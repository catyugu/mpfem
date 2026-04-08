# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 从难度最低，收益最高的部分开始，如果有些任务过于困难，你可以选择性放弃。
* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

这是一个非常典型的科学计算/有限元软件中的表达式系统设计。目前的实现中存在几个明显的**设计反模式（Anti-patterns）**和**性能陷阱**，尤其是在高性能计算（FEM积分点计算）的上下文中：

### 🚨 当前代码的四大核心问题分析

1.  **双重图/AST反模式与过度设计 (Overengineering)：**
    * `ExpressionParser` 将字符串解析为 `AstNode`，然后将其展平为基于栈/寄存器的字节码（`Instruction`, `OpCode`, `FlattenedProgram`）。
    * `VariableManager` 维护了另一个级别的 DAG（`VariableNode`）。
    * **结果**：一个简单的数学表达式被解析两次、包装两次。为了执行一个简单的 `+`，系统要通过 DAG 节点，再进入一个模拟的基于 `switch-case` 的虚拟机 (`runProgram`)，极度臃肿且没有必要。
2.  **紧密循环中的动态内存分配 (Memory Allocation in Tight Loops)：**
    * 在 `RuntimeExpressionNode::evaluateBatch` 中，每次调用（FEM中的每个批次）都会分配 `std::vector<TensorValue> localScratchpad` 和 `pointInputs`。
    * 在虚拟机的 `runProgram` 中，每次调用（FEM中的每个积分点！）都会分配 `std::vector<TensorValue> registers`。
    * **结果**：这将导致灾难性的性能瓶颈。
3.  **巨大的 `switch-case` 与扩展性差 (Violation of Open-Closed Principle)：**
    * `BuiltinUnary` 和 `BuiltinBinary` 枚举硬编码了所有支持的数学函数。如果想添加 `sinh`，需要改动词法分析、AST定义、Shape推导、OpCode定义以及虚拟机的 `executeUnary`。
4.  **运行时多态开销 (`std::visit` 滥用)：**
    * `TensorValue` 每次加减乘除都在运行时通过 `std::visit` 进行类型分发。虽然很安全，但在已知类型的 DAG 中，这是极大的性能浪费。

### 步骤化重构方案（破坏式、自动DAG+解析统一）

我们的核心思路是：**让 AST 就是 DAG，干掉中间的字节码虚拟机；静态分配内存；引入操作符注册表。**

#### 步骤 1：统一 AST 与 DAG，废除字节码虚拟机
**目标**：删除所有的 `Instruction`、`OpCode`、`FlattenedProgram`。让 `ExpressionParser` 直接生成 `VariableNode` 树。

**重构动作：**
我们将 `VariableNode` 定义为真正的计算图节点。

```cpp
// 1. 新的抽象 VariableNode (代替旧的 VariableNode 和 AstNode)
class ExprNode {
public:
    virtual ~ExprNode() = default;
    virtual TensorShape shape() const = 0;
    
    // 取消 evaluateBatch 中每次传递 std::span 作为目标的做法
    // 改为节点自带输出缓存（为后续消除 new 做准备）
    virtual void evaluateBatch(const EvaluationContext& ctx) = 0;
    virtual const std::vector<TensorValue>& getOutput() const = 0;
    
    virtual void setBatchSize(size_t size) = 0;
    virtual std::vector<std::shared_ptr<ExprNode>> dependencies() const { return {}; }
};

// 2. 基础常量节点
class ConstantNode final : public ExprNode {
    TensorValue value_;
    std::vector<TensorValue> output_;
public:
    explicit ConstantNode(TensorValue v) : value_(std::move(v)) {}
    TensorShape shape() const override { return value_.shape(); }
    void setBatchSize(size_t size) override { output_.assign(size, value_); } // 预计算
    void evaluateBatch(const EvaluationContext& ctx) override { /* 无需操作 */ }
    const std::vector<TensorValue>& getOutput() const override { return output_; }
};
```
*验证：* 代码大幅度减少，删除了上千行的 VM 机制代码。

#### 步骤 2：消除所有动态内存分配（预分配工作区）
**目标**：解决在 `evaluateBatch` 和 `runProgram` 中不断的 `std::vector` 分配。

**重构动作：**
DAG 在编译期（`compileGraph`）获知批处理大小（或设置一个合理的最大值），每个节点在内部持有自己的输出缓冲区。

```cpp
// 3. 通用二元操作节点工厂模式
template <typename OpFunc>
class BinaryOpNode final : public ExprNode {
    std::shared_ptr<ExprNode> lhs_, rhs_;
    std::vector<TensorValue> output_;
    TensorShape shape_;
    OpFunc op_;
public:
    BinaryOpNode(std::shared_ptr<ExprNode> lhs, std::shared_ptr<ExprNode> rhs, OpFunc op, TensorShape shape)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op_(std::move(op)), shape_(std::move(shape)) {}

    TensorShape shape() const override { return shape_; }
    std::vector<std::shared_ptr<ExprNode>> dependencies() const override { return {lhs_, rhs_}; }

    void setBatchSize(size_t size) override {
        output_.resize(size);
        lhs_->setBatchSize(size);
        rhs_->setBatchSize(size);
    }

    void evaluateBatch(const EvaluationContext& ctx) override {
        lhs_->evaluateBatch(ctx);
        rhs_->evaluateBatch(ctx);
        const auto& l_out = lhs_->getOutput();
        const auto& r_out = rhs_->getOutput();
        // 无内存分配的紧密循环
        for(size_t i = 0; i < output_.size(); ++i) {
            output_[i] = op_(l_out[i], r_out[i]); 
        }
    }
    const std::vector<TensorValue>& getOutput() const override { return output_; }
};
```
*验证：* 在一个物理场更新循环中，除了最开始的一次 `resize`，不再有任何 heap allocation 发生。

#### 步骤 3：引入注册表模式 (Registry)，根除硬编码和冗长的 Parser
**目标**：让解析器（Parser）不再依赖巨大的 `switch-case` 来匹配 `sin`, `cos` 等函数，彻底消除 `BuiltinUnary` 的反模式。

**重构动作：**
设计一个全局或局部的操作符注册表，Parser 在遇到函数或算符时直接查表生成节点。

```cpp
using NodeFactory = std::function<std::shared_ptr<ExprNode>(const std::vector<std::shared_ptr<ExprNode>>&)>;

class OpRegistry {
    std::unordered_map<std::string, NodeFactory> factories_;
public:
    static OpRegistry& instance() {
        static OpRegistry reg;
        return reg;
    }

    void registerOp(const std::string& name, NodeFactory factory) {
        factories_[name] = std::move(factory);
    }

    std::shared_ptr<ExprNode> createNode(const std::string& name, const std::vector<std::shared_ptr<ExprNode>>& args) {
        auto it = factories_.find(name);
        if (it == factories_.end()) MPFEM_THROW(ArgumentException, "Unknown op: " + name);
        return it->second(args);
    }
};

// 注册时 (例如在系统初始化时)
OpRegistry::instance().registerOp("add", [](const std::vector<std::shared_ptr<ExprNode>>& args) {
    // 可以在这里做 Shape 检查推导
    TensorShape outShape = inferAddShape(args[0]->shape(), args[1]->shape());
    return std::make_shared<BinaryOpNode<decltype(&mpfem::add)>>(args[0], args[1], mpfem::add, outShape);
});

OpRegistry::instance().registerOp("sin", [](const std::vector<std::shared_ptr<ExprNode>>& args) {
    return std::make_shared<UnaryOpNode<...>>(args[0], ...);
});
```
*验证：* `ExpressionParser::parseFunction` 被简化为提取名字和参数，然后调用 `OpRegistry::createNode`。添加新函数只需在 `.cpp` 文件中注册，完全符合开闭原则（OCP）。

#### 步骤 4：基于类型的 JIT 级别优化 (消除内部 `std::visit`)
**目标**：既然在 `ExpressionParser` 中可以通过 `TensorShape` 推导出返回类型，我们可以在**构建 DAG 节点时**，绑定对应类型的特定 Lambda，从而在 `evaluateBatch` 的 `for` 循环中消除 `std::visit` 开销。

**重构动作：**
对 `BinaryOpNode` 进一步特化。

```cpp
// 注册时，基于输入 Shape 静态绑定类型特定的计算逻辑
OpRegistry::instance().registerOp("add", [](const std::vector<std::shared_ptr<ExprNode>>& args) {
    auto s0 = args[0]->shape();
    auto s1 = args[1]->shape();
    
    if (s0.isScalar() && s1.isScalar()) {
        auto op = [](const TensorValue& a, const TensorValue& b) -> TensorValue {
            // 绕过泛型的 std::visit 操作符，直接提取 double
            return TensorValue::scalar(a.asScalar() + b.asScalar());
        };
        return std::make_shared<BinaryOpNode<decltype(op)>>(args[0], args[1], op, TensorShape::scalar());
    }
    // 其它类型同理 (Vector + Vector 等)...
});
```
*验证：* 在 `evaluateBatch` 循环中，由于 Lambda 内部已经直接调用了 `asScalar()` (强制读取)，从而避开了 `TensorValue` 中臃肿的 `std::visit` 类型推断逻辑。这使得数值计算的速度接近原生 C++。

#### 步骤 5：统一 DAG 与 VariableManager
**目标**：彻底将 `VariableManager` 和 `ExpressionParser` 融合。

**重构动作：**
`VariableManager` 其实就是一个统一的 `SymbolTable`。
```cpp
class VariableManager {
    std::unordered_map<std::string, std::shared_ptr<ExprNode>> symbols_;
    std::vector<std::shared_ptr<ExprNode>> executionPlan_; // 拓扑排序后的节点
    
public:
    // 解析表达式时，Parser 直接去查 symbols_
    void registerExpression(const std::string& name, const std::string& expr) {
        ExpressionParser parser(this); // 传入自身作为符号解析器
        auto rootNode = parser.parse(expr);
        symbols_[name] = rootNode;
    }
    
    // 外部可以直接注册场变量
    void registerGridFunction(const std::string& name, const GridFunction* field) {
        symbols_[name] = std::make_shared<GridFunctionNode>(field);
    }
};
```

### 总结
通过上述 5 个步骤的破坏式重构：
1. **接口一致性**：整个系统只有一个 `ExprNode`，即是 AST 节点也是计算图 DAG 节点。
2. **零运行时分配**：通过 `setBatchSize` 预先在各个节点分配好 `std::vector<TensorValue>` 内存块，计算阶段 0 分配。
3. **极简代码**：抛弃了过度设计的基于栈的虚拟机，节省了大量的 `OpCode` 维护工作。
4. **极致性能**：通过 DAG 编译期的类型推断绑定特化 Lambda，消除了 `std::visit` 带来的运行时类型推断开销。
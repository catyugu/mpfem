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

通过对你提供的代码进行深入分析，我发现虽然代码在尽力避免堆分配（例如使用 `std::variant` 实现 `TensorValue`），但在**表达式解析接口**、**DAG 节点抽象**以及**变量图管理**方面存在一些典型的设计反模式（Anti-patterns）、冗余和职责泄漏。

按照你的要求（宁可破坏向后兼容也要保持简洁高效、统一自动DAG和表达式解析系统），我们将分 4 个步骤进行“破坏式”重构。

---

### 发现的设计反模式与核心问题

1.  **类型擦除与接口不一致 (Type Erasure & Inconsistency)**：
    * `ExpressionProgram` 返回 `TensorValue`，但 `VariableNode::evaluateBatch` 却要求输出到 `std::span<Real>`。这破坏了 `TensorValue` 建立的强类型系统，导致图节点在求值时需要手动处理内存展平/反展平，极易出错且不支持复杂的张量图计算。
2.  **上帝对象与职责泄漏 (God Object & Leaky Abstraction)**：
    * `VariableManager` 内部竟然包含了 `ensureGradientNode` 和 `gridFunctions_`。DAG 管理器不应该知道“梯度”或“网格函数”是什么，这属于物理/有限元业务逻辑，严重违反了**开闭原则 (OCP)**。
3.  **冗余接口 (Redundant Interfaces)**：
    * `ExpressionProgram` 提供了 `std::span<const TensorValue>` 和 `std::span<const Real>` 两个求值接口，纯属多余。

### 步骤化重构方案

#### 第一步：统一数据流通量（消除类型擦除与冗余 API）

**目标**：强制整个表达式和 DAG 系统中只流动 `TensorValue`，移除所有降级到 `Real` 裸指针/Span 的冗余接口。

**修改 `src/expr/expression_parser.hpp`**：
移除接受 `Real` 的重载，保持唯一事实来源。
```cpp
// 重构后的 ExpressionProgram
class ExpressionProgram {
public:
    // ... 构造/析构保持不变 ...
    bool valid() const;
    TensorShape shape() const;
    const std::vector<std::string>& dependencies() const;
    
    // [破坏性重构] 移除 TensorValue evaluate(std::span<const Real>) const;
    // 强制使用统一的 TensorValue 接口
    TensorValue evaluate(std::span<const TensorValue> values) const;
};
```

**修改 `src/expr/variable_graph.hpp` (节点抽象)**：
将 `evaluateBatch` 的输出从 `std::span<Real>` 改为 `std::span<TensorValue>`。
```cpp
// 重构后的 VariableNode
class VariableNode {
public:
    virtual ~VariableNode() = default;

    virtual TensorShape shape() const = 0;

    // [破坏性重构] 拒绝退化为 Real 数组，直接使用 TensorValue 数组进行 Batch 写入
    // 这消除了图节点内部手动计算 stride 的痛苦，且完美兼容常数、向量、矩阵
    virtual void evaluateBatch(const EvaluationContext& ctx, 
                               std::span<TensorValue> dest) const = 0;

    virtual bool isConstant() const { return false; }
    virtual std::vector<const VariableNode*> dependencies() const { return {}; }
};
```

#### 第二步：引入 `EvaluationWorkspace` 解决计算内存分配

**目标**：DAG 执行时需要存储中间变量，目前代码没有体现这些中间变量存在哪里。引入一个统一的工作区（Workspace），在一次评估中重用内存。

**新增类 (可在 `variable_graph.hpp` 中)**：
```cpp
namespace mpfem {
    // 管理一次 Batch 评估中所有节点的临时/结果数据
    class EvaluationWorkspace {
    public:
        void allocate(size_t batchSize, const std::vector<const VariableNode*>& plan) {
            batchSize_ = batchSize;
            memory_.clear();
            memory_.resize(plan.size() * batchSize);
            
            nodeOffsets_.clear();
            size_t offset = 0;
            for (const auto* node : plan) {
                nodeOffsets_[node] = offset;
                offset += batchSize;
            }
        }

        std::span<TensorValue> getBuffer(const VariableNode* node) {
            return {memory_.data() + nodeOffsets_.at(node), batchSize_};
        }

        std::span<const TensorValue> getBuffer(const VariableNode* node) const {
            return {memory_.data() + nodeOffsets_.at(node), batchSize_};
        }

    private:
        size_t batchSize_ = 0;
        std::vector<TensorValue> memory_; 
        std::unordered_map<const VariableNode*, size_t> nodeOffsets_;
    };
}
```

#### 第三步：剥离 `VariableManager` 的业务逻辑（纯化 DAG 管理）

**目标**：把 `VariableManager` 变成一个纯粹的、仅关心图拓扑排序和节点拥有的容器。把 `GridFunction` 和梯度逻辑踢出这个类。

**修改 `src/expr/variable_graph.hpp` (`VariableManager`)**：
```cpp
class VariableManager {
public:
    using NodeStore = std::unordered_map<std::string, std::unique_ptr<VariableNode>>;

    VariableManager() = default;
    ~VariableManager() = default;

    // 核心 API 1：接管节点所有权 (所有特殊节点通过此接口注入)
    void registerNode(std::string name, std::unique_ptr<VariableNode> node);

    // 核心 API 2：基于字符串表达式自动构建 RuntimeExpressionNode
    void registerExpression(std::string name, std::string expression);

    const VariableNode* get(std::string_view name) const;

    // 编译图结构，生成拓扑排序的执行计划，并分配 Workspace
    void compileGraph();

    // 执行求值
    void evaluate(const EvaluationContext& ctx, EvaluationWorkspace& workspace) const;

private:
    void clearExecutionPlan();
    void dfsVisit(const VariableNode* node,
                  std::unordered_map<const VariableNode*, int>& marks,
                  std::vector<const VariableNode*>& ordered) const;

    NodeStore nodes_;
    std::vector<const VariableNode*> executionPlan_;
    bool graphDirty_ = true;

    // [破坏性重构] 移除了以下业务逻辑，它们应该由外部 Factory 包装后调用 registerNode 注入
    // void ensureGradientNode(std::string_view symbol);
    // std::unordered_map<std::string, const GridFunction*> gridFunctions_;
    // void registerGridFunction(...);
    // void registerExternalSource(...);
};
```

#### 第四步：实现统一的 `RuntimeExpressionNode`（闭环验证）

现在，因为图管理纯化了，且输入输出都是统一的 `TensorValue`，我们终于可以写出一个极度干净、支持任何张量表达式的解析节点实现：

```cpp
// 假设这是在 variable_graph.cpp 中的实现
class RuntimeExpressionNode : public VariableNode {
public:
    RuntimeExpressionNode(ExpressionParser::ExpressionProgram prog, 
                          std::vector<const VariableNode*> deps)
        : program_(std::move(prog)), deps_(std::move(deps)) {}

    TensorShape shape() const override { return program_.shape(); }
    std::vector<const VariableNode*> dependencies() const override { return deps_; }

    void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
        size_t batchSize = ctx.physicalPoints.size();
        
        // 准备单个点的输入缓冲区
        std::vector<TensorValue> pointInputs(deps_.size());
        
        // 注意：这里的实现为了展示逻辑结构。
        // 在实际的高性能框架中，Workspace 应该直接传给 evaluateBatch，
        // 从而可以执行向量化求值，而不是每次在循环里收集依赖。
        
        for (size_t i = 0; i < batchSize; ++i) {
            // 这里假设外部的 workspace 已经帮依赖节点求好值了
            // 实际工程中，你需要将 workspace 传进来以获取依赖的值。
            // 例如： pointInputs[j] = workspace.getBuffer(deps_[j])[i];
            
            dest[i] = program_.evaluate(pointInputs);
        }
    }

private:
    ExpressionParser::ExpressionProgram program_;
    std::vector<const VariableNode*> deps_;
};
```

### 重构带来的好处：

1. **接口极其一致**：从 `TensorValue` 的变体，到 `ExpressionProgram`，再到 `VariableNode` 的 IO，全量统一为 `TensorValue`。不再有 `Real` 和 `TensorValue` 的混用。
2. **纯粹的 DAG (SoC)**：`VariableManager` 现在只负责一件事——有向无环图的拓扑排序和执行。什么是“场变量”、什么是“梯度”，变成了一个个实现了 `VariableNode` 的派生类，由外部工厂创建后注册进去。
3. **消除隐式循环依赖**：原先的 `Manager` 强依赖了 `GridFunction`，重构后它只依赖抽象的 `VariableNode`，`GridFunctionNode` 可以在 `fe/` 或 `physics/` 目录中单独实现并注入，解除了 `expr` 模块对 `fe` 模块的逆向依赖。

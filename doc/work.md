# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 删除冗余的成员变量、接口等。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

### 🚨 核心反模式诊断 (Anti-patterns)

#### 1. 性能杀手：在最内层积分点循环中动态分配 `std::unordered_map`
**位置**：`variable_graph.cpp` 中的 `RuntimeScalarExpressionNode::evaluateBatch` 和 `makeValueMap`
**问题**：
```cpp
for (size_t i = 0; i < n; ++i) { // 遍历物理/积分点
    // 【致命反模式】：每次循环都新建一个哈希表并插入所有变量！
    const std::unordered_map<std::string, double> values = makeValueMap(...); 
    dest[i] = program_.evaluate(values);
}
```
在 FEM 组装中，这会被调用数百万次。动态内存分配（`new`）、字符串哈希（Hash）、红黑树/哈希表插入的开销，将远远超过实际数学表达式（如 `+ - * /`）计算的开销（慢几个数量级）。

#### 2. 标量化的 AST 递归求值 (Scalar Recursive AST Evaluation)
**位置**：`expression_parser.cpp` 中的 `evalAstNode`
**问题**：对每一个物理点，AST 都要进行一次深度的递归函数调用树遍历，并且伴随着巨大的 `switch-case` 开销和指针跳转。这种标量计算无法利用现代 CPU 的 SIMD/向量化指令。

#### 3. 割裂的物理场与变量系统 (Fragmented Field vs. Variable System)
**位置**：`field_values.hpp` vs `variable_graph.hpp`
**问题**：`FieldValues` 孤立存在，只是一个管理 `GridFunction` 的容器。而 `VariableManager` 只是解析文本表达式。在多物理场中，**位移场、温度场本身也应该只是 DAG 中的一个节点**。现在的设计导致表达式无法直接、透明地获取和依赖实际的物理场。

#### 4. 矩阵表达式只是 9 个孤立的标量表达式
**位置**：`ExpressionParser::MatrixProgram`
**问题**：如果用户写了 `{{ a*b, 0, 0 }, { 0, a*b, 0 }, ...}`，`a*b` 会被解析并计算多次。缺乏原生的 Tensor/Vector AST 支持。

#### 5. 硬编码的“内建符号” (Hardcoded Built-ins)
**位置**：`BuiltInSymbolKind` (`x`, `y`, `z`, `t`)
**问题**：在纯粹的 DAG 系统中，不应该有“特殊”的变量。时间 $t$ 只是一个提供单值输出的常量节点，空间坐标 $(x, y, z)$ 只是提供几何映射的普通变量节点。硬编码破坏了一致性。

---

### 🛠️ 破坏式重构蓝图 (Refactoring Blueprint)

为了实现您“**完全用自动 DAG + 表达式系统取代**”的意图，建议彻底抛弃目前的字符串+哈希表运行时绑定，转向**编译期（Setup阶段）图链接 + 运行时平坦数组/向量化计算**。

#### 重构步骤 1：统一所有实体为 `VariableNode` (DAG 核心化)
放弃 `FieldValues` 独立管理的思路。时间和空间坐标也是 Node。
一切皆 `Node`。

```cpp
// 基础抽象：批量（向量化）求值的 DAG 节点
class VariableNode {
public:
    virtual ~VariableNode() = default;
    
    virtual VariableShape shape() const = 0;
    
    // 强制使用向量化接口：一次性计算所有积分点/物理点
    // 输入 ctx 包含点信息，输出直接写入 dest
    virtual void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const = 0;
    
    // 返回该节点依赖的其他节点
    virtual std::vector<const VariableNode*> dependencies() const = 0;
};
```

**实现各种具体的 Node：**
* `TimeNode`：无依赖，在 `evaluateBatch` 中用 `ctx.time` 填充 `dest`。
* `CoordinateNode`：无依赖，用 `ctx.physicalPoints` 填充 `dest`。
* `GridFunctionNode`：依赖几何映射，根据 `ctx.referencePoints` 和 `GridFunction` 插值出物理值。
* `ExpressionNode`：依赖其子节点（从文本解析而来）。

#### 重构步骤 2：表达式系统脱离字符串绑定 (Flat Memory Model)
`ExpressionParser` 解析后的 `ScalarProgram` **不应该**知道变量名叫什么，它只需要知道**变量在输入数组中的索引**。

**旧设计**：`evaluate(unordered_map<string, double>)`
**新设计**：`evaluate(const double* inputs)`

```cpp
// 重构后的标量程序
class ScalarProgram {
public:
    // 运行时极速求值：不按点求值，而是直接传入批量的依赖数据，输出批量结果
    // input_blocks 存放了其所有依赖节点的 evaluateBatch 结果
    void evaluateBatch(const std::vector<std::span<const double>>& input_blocks, 
                       std::span<double> dest) const;
};
```

#### 重构步骤 3：AST 向量化求值 (Vectorized AST)
修改 `AstNode` 的执行逻辑，不要递归计算单精度/双精度值，而是直接在数组上操作。

```cpp
// 在 expression_parser.cpp 内部
void evalAstNodeBatch(const AstNode& node, 
                      const std::vector<std::span<const double>>& inputs, 
                      std::span<double> dest) {
    switch (node.kind) {
        case AstNode::Kind::Variable: {
            // 在编译图的阶段，已经把变量名映射成了 inputs 数组的索引
            const auto& input_data = inputs[node.variable_index];
            std::copy(input_data.begin(), input_data.end(), dest.begin());
            break;
        }
        case AstNode::Kind::Add: {
            // 分配临时缓冲区（或者利用内存池）
            std::vector<double> lhs_val(dest.size());
            std::vector<double> rhs_val(dest.size());
            evalAstNodeBatch(*node.lhs, inputs, lhs_val);
            evalAstNodeBatch(*node.rhs, inputs, rhs_val);
            for(size_t i=0; i<dest.size(); ++i) {
                dest[i] = lhs_val[i] + rhs_val[i];
            }
            break;
        }
        // ...
    }
}
```
*(注：如果想极致优化，可以将 AST 展平为**字节码 (Bytecode)**，例如转成逆波兰表达式 RPN 放在一个 `std::vector<Instruction>` 中执行，彻底消除递归。)*

#### 重构步骤 4：VariableManager 作为计算图调度器 (DAG Scheduler)
`VariableManager` 的核心任务变成：
1. 注册所有的 Node（包括时间、空间、物理场、表达式）。
2. `compileGraph()`：建立拓扑排序。**在这里完成所有字符串名字到数组索引的绑定**。

```cpp
void VariableManager::compileGraph() {
    // 1. 拓扑排序 (你已经实现了 dfsVisit，非常好)
    executionPlan_ = topologicalSort(nodes_);

    // 2. 建立绑定关系！
    for (auto* node : executionPlan_) {
        if (auto* exprNode = dynamic_cast<RuntimeScalarExpressionNode*>(node)) {
            // 查找表达式依赖的名字，将它们转换为具体的 Node 指针或执行上下文索引
            exprNode->bindDependencies(this);
        }
    }
}
```

#### 重构步骤 5：求值管道 (Evaluation Pipeline)
当求解器请求组装矩阵，需要求值时，工作流非常清晰，没有任何内存分配：

```cpp
void evaluateGraph(const EvaluationContext& ctx) {
    // 假设本批次有 100 个积分点
    size_t num_points = ctx.physicalPoints.size();
    
    // 为图中的每个节点分配一块连续的临时内存（可以使用线程局部的 Memory Pool 防止重复 new）
    MemoryPool pool(num_points * executionPlan_.size()); 

    for (const VariableNode* node : executionPlan_) {
        std::span<double> dest = pool.allocate(num_points * node->vdim());
        
        // 每个节点会自动去读它依赖节点的内存块，并写到自己的 dest 中
        node->evaluateBatch(ctx, dest); 
    }
}
```

---

### 💡 总结与建议

您当前代码最大的痛点在于**将 "描述（文本/字符串）" 泄露到了 "最底层的数值求值（Evaluation）" 环节**。

破坏式重构的核心法则：
1. **彻底分离编译期和运行期**：所有根据 `std::string` 查找变量的行为，**必须且只能**发生在 `VariableManager::compileGraph()` 中。
2. **消灭点级（Point-wise）接口**：强迫所有的节点使用 `evaluateBatch(std::span<double>)`，这不仅消除了循环内的临时变量创建，更自动为未来的 AVX/SIMD 向量化铺平了道路。
3. **融合 Field 与 Variable**：`GridFunction`（或者包装它的代理类）本身就是 DAG 中的一个 Source Node。这样无论是 PDE 弱形式还是边界条件的表达式，都可以统一地写成 `expr("Temperature^4 * 5.67e-8")`，系统在图排序时自动将温度场节点链接给表达式节点。
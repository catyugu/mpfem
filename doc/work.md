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

针对您提供的 `mpfem` 变量系统与表达式解析器的代码，通过分析发现存在以下主要问题：

### 现状分析与反模式识别

1.  **类型处理不统一**：`AstNode` 的 `Kind` 过于繁琐（区分 `VectorLiteral` 和 `MatrixLiteral`），未能充分利用已经存在的 `TensorValue` 调度能力。
2.  **求值效率低**：`RuntimeExpressionNode` 在 `evaluateBatch` 中使用 `thread_local` 的 `unordered_map` 和大量的 `std::vector` 分配作为临时缓冲区，这会产生巨大的运行时开销。
3.  **循环依赖隐患**：`VariableManager` 在 `define` 时直接调用 `makeExpressionNode`，而 `makeExpressionNode` 又需要查看 `nodes_` 的状态。这种逻辑耦合使得图的构建和验证变得混乱。

---

### 重构步骤建议

我们将采取“破坏式重构”，目标是建立一个纯粹的、强类型的、基于张量的自动 DAG 系统。

#### 第一步：统一 AST 节点与张量字面量 (实现可编译)
**目标**：将所有字面量（标量、向量、矩阵）统一为 `BracketLiteral`。

* **操作**：
    * 重构 `AstNode`：将 `VectorLiteral` 和 `MatrixLiteral` 合并到 `Literal` 类型，并在解析期就决定其 `TensorValue` 内容。
    * 修改 `parseBracketLiteral`，使其直接利用 `TensorValue::matrix3` 或 `vector` 构造函数生成常量节点。

#### 第二步：表达式解析器降级与算子抽象 (实现可执行)
**目标**：将原本在解析器中硬编码的 `inferShape` 和 `evaluateAst` 逻辑下放到 `TensorValue` 运算符中。

* **操作**：
    * 利用 `src/core/tensor_value.hpp` 中已有的 `operator+`, `operator*`, `dot`, `sym` 等重载函数。
    * `AstNode` 不再存储复杂的求值逻辑，而是存储一个 `std::function` 或算子枚举。
    * **代码精简**：解析阶段直接根据注册的 `registeredShapes` 校验维度，如果不匹配直接抛出异常，不再在运行时进行复杂的类型检查。

#### 第三步：重构变量图的高效求值引擎 (关键重构)
**目标**：消除 `evaluateBatch` 中的动态内存分配，改为预分配缓冲区。

* **操作**：
    * 在 `VariableManager::compile()` 阶段，不仅计算拓扑排序（`executionPlan_`），还要计算每个节点所需的缓冲区大小。
    * **内存池化**：为整个 `VariableManager` 分配一块连续的 `TensorValue` 缓冲区。
    * 每个 `VariableNode` 在编译后获得该缓冲区的 `offset`。
    * `evaluateBatch` 变为：`void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> globalBuffer)`。这将运行时复杂度从 $O(N \cdot \text{MapLookup})$ 降至 $O(N)$ 指针偏移。

#### 第四步：解耦变量定义与图构建 (接口统一)
**目标**：使 `VariableManager` 成为纯粹的声明式接口。

* **操作**：
    * `define(name, expr)` 仅存储原始字符串。
    * 新增 `validate()` 阶段：检查未定义的符号、循环依赖。
    * `compile()` 阶段：一次性解析所有表达式，构建完整的 `RuntimeExpressionNode` 森林。
    * 这样可以避免在 `define` 每一个变量时都去解析一次 `registeredShapes`。

---

### 核心重构代码示例 (基于 Step 3 & 4)

重构后的执行计划逻辑应类似于：

```cpp
// 在 VariableManager.cpp 中
void VariableManager::compile() {
    // 1. 拓扑排序 (现有逻辑)
    // 2. 静态分配优化：
    size_t totalBufferSize = 0;
    for (auto* node : executionPlan_) {
        // 每个节点在 globalBuffer 中分配其存储空间
        node->setBufferOffset(totalBufferSize);
        totalBufferSize += numPhysicalPoints; // 假设批量大小固定或按需重分配
    }
    // 3. 预解析所有表达式程序
}

void VariableManager::evaluateAll(const EvaluationContext& ctx) const {
    // 严格按拓扑序执行，不再需要递归调用 evaluateBatch
    for (auto* node : executionPlan_) {
        node->execute(ctx, globalBuffer_); 
    }
}
```

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

这是一个非常典型的“半重构状态”代码库。代码中显然正在经历从“传统的硬编码数值计算/回调系统”向“现代的基于DAG（有向无环图）的表达式系统”过渡的阵痛期。

你当前代码中最大的**设计反模式**是：**双轨制（Dual-System）与生命周期碎片化**。虽然引入了 `VariableManager` 和 `ExpressionParser`，但大量的物理场逻辑仍在绕过它，采用硬编码的 C++ 节点（如 `JouleHeatNode`）和复杂的回调（如 `GraphRuntimeResolvers`）。

以下是具体识别出的反模式，以及彻底转向 **纯DAG+表达式解析变量系统** 的破坏式重构步骤。

### 🚨 识别出的设计反模式

1. **生命周期碎片化 (Scattered Lifecycle)**: `Problem` 类中同时存在 `globalVariables_`、`expressionManagers_`（存储临时变量管理器的 `vector`）、`ownedNodes`（手动管理的 C++ 节点）以及 `domainScalarNodes/domainMatrixNodes` 缓存。这让“谁拥有这个变量”变得极其混乱。
2. **破坏开闭原则的硬编码解析 (Hardcoded Symbol Resolution)**: 在 `physics_problem_builder.cpp` 中，`classifyRuntimeField` 函数硬编码了 `"T"` 和 `"V"`。如果未来加入流体力学求解器（需要 `"U"`, `"P"`），你必须修改构建器代码。
3. **退化的 C++ 硬编码表达式节点 (Hardcoded Expression Nodes)**: `JouleHeatNode`、`ThermalExpansionStressNode` 以及 `HeatTransferSolver` 内部的 `ProductScalarNode`。既然已经有了表达式解析器，像 $\rho \cdot C_p$ 这样的逻辑应该直接解析为表达式，而不是在 C++ 中手动拼接 AST（抽象语法树）。
4. **冗余的 Domain 缓存 (Redundant Domain Caching)**: `DomainPropertyKey` 和其相关的 Map 完全是多余的。DAG 本身的作用就是复用节点，基于 Domain 的差异应该通过在统一的全局 DAG 中注册不同的变量名（如 `density_dom1`）来解决。


---

### 🛠️ 步骤化破坏式重构方案

每一步都保证系统处于可编译、可运行的完整状态，宁可采用破坏性修改也不保留向后兼容的包袱。

#### 第一步：引入场自注册机制，废弃硬编码解析器
**目标**：消除 `GraphRuntimeResolvers` 和 `classifyRuntimeField`，让求解器将其结果作为 DAG 节点直接注入到全局变量图中。

1. **新增基础节点**：在 `variable_graph.hpp` 中添加 `GridFunctionNode`。
   ```cpp
   class GridFunctionNode final : public VariableNode {
   public:
       explicit GridFunctionNode(const GridFunction* field) : field_(field) {}
       VariableShape shape() const override { return VariableShape::Scalar; } // 或根据场决定
       std::pair<int, int> dimensions() const override { return {1, 1}; }
       void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override {
           // 利用 ctx.transform 和 ctx.referencePoints 直接从 field_ 中采样计算
       }
   private:
       const GridFunction* field_;
   };
   ```
2. **求解器自注册**：修改 `HeatTransferSolver::initialize` 和 `ElectrostaticsSolver::initialize`，要求传入全局的 `VariableManager&`。
   ```cpp
   // 在求解器初始化完毕后：
   globalVarManager.registerNode("T", std::make_unique<GridFunctionNode>(&this->field()));
   ```
3. **删减冗余代码**：
   - 彻底删除 `physics_problem_builder.cpp` 中的 `classifyRuntimeField`、`resolveRuntimeField` 和 `makeRuntimeExpressionResolvers`。
   - `VariableManager::registerScalarExpression` 的签名移除 `GraphRuntimeResolvers` 参数，因为此时 `"T"` 已经是 DAG 中的一个合法前置节点。

#### 第二步：统一全局 DAG，消灭临时管理器与域缓存
**目标**：让 `VariableManager` 成为唯一的事实来源（Single Source of Truth）。

1. **重构 Problem 类**：
   - **删除** `domainScalarNodes`、`domainMatrixNodes`。
   - **删除** `expressionManagers_`（极大的反模式，不应该为每个小表达式创建一个 Manager）。
   - **删除** `ownedNodes`。
   - 保留唯一的 `VariableManager globalVariables_;`。
2. **规范化命名方案**：在 `physics_problem_builder.cpp` 中，当需要获取某个 Domain 的属性时，利用名称组合而不是 Map 缓存。
   ```cpp
   // 替代原有的 requireDomainScalarNode
   const VariableNode* requireDomainScalarNode(Problem& problem, int domainId, std::string_view property) {
       std::string nodeName = std::string(property) + "_dom" + std::to_string(domainId);
       if (auto* node = problem.globalVariables_.get(nodeName)) return node;
       
       const std::string& expr = problem.materials.scalarExpressionByDomain(domainId, property);
       problem.globalVariables_.registerScalarExpression(nodeName, expr);
       return problem.globalVariables_.get(nodeName);
   }
   ```
   **编译验证**：此时所有的变量和表达式都位于同一个 DAG 中，并且按名称索引，无需外部缓存。

#### 第三步：用动态解析替换局部硬编码 AST (消灭 `ProductScalarNode`)
**目标**：清理具体求解器中手写的 AST 组合。

1. **修改 HeatTransferSolver**：
   打开 `heat_transfer_solver.cpp`，定位到 `ProductScalarNode` 类。
   **删除它**。
2. **重构组装逻辑**：在 `HeatTransferSolver::assembleMassMatrix` 中，我们不需要自己写乘法节点，而是委托给 DAG。
   在给 `HeatTransferSolver` 赋值时，不传入指针，而是传入变量名称。或者让 Builder 生成乘积表达式：
   ```cpp
   // 在 builder 中
   std::string rhoName = "density_dom" + std::to_string(domainId);
   std::string cpName  = "heatcapacity_dom" + std::to_string(domainId);
   std::string massPropName = "mass_prop_dom" + std::to_string(domainId);
   problem.globalVariables_.registerScalarExpression(massPropName, rhoName + " * " + cpName);
   
   problem.heatTransfer->setMassProperties({domainId}, problem.globalVariables_.get(massPropName));
   ```
   *注意：这一步不仅使得代码行数锐减，而且如果将来表达式系统增加了常数折叠（Constant Folding）或公共子表达式消除优化，这些逻辑将自动受益。*

#### 第四步：抽象场操作符，消灭 C++ 物理耦合节点 (JouleHeating / ThermalExpansion)
**目标**：删除 `JouleHeatNode` 和 `ThermalExpansionStressNode`。这是最硬核的解耦。

1. **引入算子节点 (Operator Nodes)**：
   当前的解析器可以解析基础的数学运算，但 FEM 中涉及到场的梯度算子（$\nabla$）。
   在 `variable_graph.hpp` 中增加 `FieldGradientNode`：
   ```cpp
   class FieldGradientNode final : public VariableNode {
       // 返回形状：VariableShape::Vector
       // 内部调用 voltageField_->gradient(...) 计算真实梯度
   };
   ```
2. **暴露到表达式空间**：
   让 `VariableManager` 或 `ExpressionParser` 支持特殊变量名，例如在注入场时：
   ```cpp
   varManager.registerNode("grad_V", std::make_unique<FieldGradientNode>(&electrostatics->field()));
   ```
3. **在 Builder 中使用纯表达式实现耦合**：
   删除 `JouleHeatNode.cpp`。在 `setupJouleHeating` 中，完全用字符串描述焦耳热：
   ```cpp
   std::string q_joule_expr = "dot(grad_V, sigma_dom" + std::to_string(domId) + " * grad_V)";
   problem.globalVariables_.registerScalarExpression("Q_joule_dom" + std::to_string(domId), q_joule_expr);
   problem.heatTransfer->setHeatSource(activeDomains, problem.globalVariables_.get("Q_joule_dom" + std::to_string(domId)));
   ```
   对于 `ThermalExpansionStressNode` 同理，将 $\alpha \cdot (T - T_{ref})$ 以及拉梅常数的计算写进一串表达式中，或者拆分成几条注册到 DAG 的中间变量中。

### 阶段性总结

经过这四步破坏性重构，你的系统将发生如下蜕变：
* **前**：Builder 需要了解热胀冷缩的公式并用 C++ 拼接 AST，维护 5 种不同生命周期的节点 Map。
* **后**：Builder 完全退化为一个 **依赖注入/接线员**。它读取 `case.xml` 的字符串，并将它们注册到全局唯一的 `VariableManager` 中。`VariableManager.compileGraph()` 负责所有的依赖排序和合法性检查。代码量预计减少 30% 到 40%，且扩展新物理场（如流体、相变）时，仅需增加几行字符串注册代码，无需增加任何新的 C++ `Node` 派生类。
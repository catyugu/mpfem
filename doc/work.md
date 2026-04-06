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

### 识别出的设计反模式 (Anti-patterns)

1. **硬编码的物理场枚举与分类（向后兼容的毒瘤）**
   * **表现**: `field_values.hpp` 中存在硬编码的 `enum class FieldId { Temperature, ElectricPotential, Displacement }`；`physics_problem_builder.cpp` 中存在 `RuntimeFieldKind`。
   * **问题**: 违反开闭原则（OCP）。新增一个物理场（如流体速度）需要修改基础架构文件。
2. **表达式系统的“外挂式”解析（循环依赖与冗余）**
   * **表现**: `GraphRuntimeResolvers` 和 `symbolBinder`。为了让表达式能读取当前的温度 `T` 或电压 `V`，使用了一个极其复杂的闭包回调系统。
   * **问题**: 物理场值（GridFunction）本来就应该是变量图（DAG）中的一等公民（`GridFunctionNode`），不应该作为“外部符号”被特殊对待。
3. **手动构建特定物理节点（违背自动 DAG 的初衷）**
   * **表现**: `JouleHeatNode` 和 `ThermalExpansionStressNode` 是手写的 C++ 类，且包含大量手写的求导和矩阵乘法逻辑。
   * **问题**: 焦耳热就是 $(\nabla V)^T \cdot \sigma \cdot \nabla V$。这本该由表达式系统和 DAG 自动处理，现在却退化回了硬编码，使得变量系统名存实亡。
4. **标量与矩阵的类型割裂（冗长代码）**
   * **表现**: `registerScalarExpression` vs `registerMatrixExpression`，`RuntimeScalarExpressionNode` vs `RuntimeMatrixExpressionNode`。
   * **问题**: 表达式程序（`ExpressionProgram`）自身已经知道其 `VariableShape`，但在外部管理者中却被强制分流，导致代码重复。
5. **巨型上帝工厂类（高耦合）**
   * **表现**: `PhysicsProblemBuilder` 的 `buildSolvers` 包含了所有物理场的初始化逻辑，且 `Problem` 类硬编码了 `electrostatics`, `heatTransfer` 等指针。

---

### 步骤化重构建议 (Step-by-Step Refactoring Plan)

目标：**一切皆节点（Everything is a Node），全自动 DAG 求值**。每一步都保证可编译、可验证。

#### Step 1: 废除 `FieldId` 枚举，实现物理场字符串注册 (依赖倒置)
**动作**: 将所有基于 Enum 的物理场标识替换为字符串。
1. 删除 `enum class FieldId` 及其 `toString` 方法。
2. 修改 `FieldValues`，将其内部的 `std::map<FieldId, FieldEntry>` 改为 `std::unordered_map<std::string, FieldEntry>`。
3. 修改 `PhysicsFieldSolver` 接口，将 `virtual FieldId fieldId() const = 0;` 改为 `virtual std::string fieldName() const = 0;`。
4. **验证**: 编译所有 solver，确保通过 `fieldValues_->current("V")` 获取电压场。不再需要为了加一个场去改核心头文件。

#### Step 2: 物理场回归 DAG 一等公民，彻底删除 `GraphRuntimeResolvers`
**动作**: 消除表达式解析时对外挂回调的依赖，将物理场直接注册到 `VariableManager`。
1. 删除 `variable_graph.hpp` 中的 `GraphExternalSymbolResolver`, `GraphExternalSymbolBinder`, `GraphRuntimeResolvers` 等结构。
2. 在每个 Solver 初始化时（如 `ElectrostaticsSolver::initialize`），主动向全局的 `VariableManager` 注册自己：
   ```cpp
   // 在 solver 初始化后立刻执行
   problem.globalVariables_.registerGridFunction("V", &this->field());
   ```
3. 修改 `RuntimeSymbolConfig buildSymbolConfig(...)` 函数：遇到符号时，直接在 `VariableManager` 的已注册节点中查找。如果找不到，报错（而不是求助 resolver）。
4. **验证**: 焦耳热或材料属性对 `T` 和 `V` 的依赖现在自然地通过 DAG 解析，不再需要恶心的回调闭包。

#### Step 3: 统一表达式节点注册（消除类型分裂）
**动作**: 让 `VariableManager` 依靠 `ExpressionProgram` 自身的元数据决定分配什么节点。
1. 删除 `registerScalarExpression` 和 `registerMatrixExpression`，合并为单一的 `void registerExpression(std::string name, std::string exprText)`。
2. 合并 `RuntimeScalarExpressionNode` 和 `RuntimeMatrixExpressionNode` 为统一的 `RuntimeExpressionNode`：
   ```cpp
   class RuntimeExpressionNode final : public VariableNode {
       // ...
       VariableShape shape() const override { return program_.shape(); }
       std::pair<int, int> dimensions() const override { 
           return (program_.shape() == VariableShape::Matrix) ? std::pair{3, 3} : std::pair{1, 1}; 
       }
       void evaluateBatch(...) const override {
           // 内部根据 shape 决定写 1 个值还是 9 个值
       }
   };
   ```
3. **验证**: `PhysicsProblemBuilder` 中不再需要判断调用哪个注册函数，只需一把梭 `registerExpression`。

#### Step 4: 引入空间梯度节点，屠杀手写物理节点 (核心破坏式重构)
**动作**: 用 DAG 节点替代 `JouleHeatNode` 和 `ThermalExpansionStressNode`。
1. 在 `variable_graph.cpp` 中新增 `GridFunctionGradientNode`：
   ```cpp
   class GridFunctionGradientNode final : public VariableNode {
       // 接受一个 GridFunction，shape 返回 Vector (或 3x1 Matrix)。
       // evaluateBatch 时调用 field_->gradient(...)
   };
   ```
2. 为 `VariableManager` 增加基础张量运算的支持（这要求你更新你的 `ExpressionParser`，使其支持向量/矩阵的内积、点积等运算；如果暂时不支持解析器层面的张量运算，可以退而求其次，在 DAG 层面实现 `TensorDotNode` 等基础算子节点）。
3. **重写焦耳热** (`setupJouleHeating`)：
   ```cpp
   // 不再使用硬编码的 JouleHeatNode
   problem.globalVariables_.registerGradient("grad_V", "V"); // 自动求导节点
   // 利用表达式系统计算焦耳热：Q = dot(grad_V, sigma * grad_V)
   problem.globalVariables_.registerExpression("JouleHeat", "dot(grad_V, sigma * grad_V)");
   ```
4. **验证**: 删除 `JouleHeatNode` 类。此时你的变量系统已经实现了真正的图计算。

#### Step 5: 剥离 `PhysicsProblemBuilder` 的硬编码装配（插件化）
**动作**: 把 `Problem` 和 Builder 变成真正的框架，对具体的物理场无感。
1. 在 `Problem` 中删除特定的成员变量：
   ```cpp
   // 删除：
   // std::unique_ptr<ElectrostaticsSolver> electrostatics;
   // std::unique_ptr<HeatTransferSolver> heatTransfer;
   // 改为：
   std::unordered_map<std::string, std::unique_ptr<PhysicsFieldSolver>> solvers_;
   ```
2. 引入一个简单的注册表（Registry 模式）或通过约定的工厂模式来实例化 Solver：
   ```cpp
   for (auto& [kind, physics] : caseDef.physics) {
       auto solver = SolverFactory::create(kind, physics.order);
       solver->initialize(*mesh, fieldValues, ...);
       solvers_[kind] = std::move(solver);
   }
   ```
3. **验证**: 现在 `PhysicsProblemBuilder` 对“静电场”、“固体力学”一无所知。新增物理场只需新增一个派生自 `PhysicsFieldSolver` 的类并在工厂注册即可。

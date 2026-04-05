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

**目前的系统处于“半自动、半硬编码”的尴尬过渡期。** 终极意图是**完全用自动DAG+表达式解析的变量系统取代原始的数值解析系统**，但当前代码中充斥着严重的设计反模式，阻碍了这一目标的实现。

以下是代码中存在的核心反模式及问题分析：

1. **反模式：硬编码的物理领域耦合（God Builder & Circular Dependency）**
   * **问题**：在 `physics_problem_builder.cpp` 中，`classifyRuntimeField` 硬编码了 `"T"` 和 `"V"`；`setupCoupling` 中手动编写了 `JouleHeatNode` 和 `ThermalExpansionStressNode` 这两个庞大的 C++ 类来计算物理公式。这完全违背了“表达式系统”的初衷。如果需要写几百行 C++ 代码来计算 $Q = \nabla V \cdot \sigma \cdot \nabla V$，那么表达式系统形同虚设。
2. **反模式：热点路径（Hot-Path）中的堆内存分配**
   * **问题**：`ProductScalarNode::evaluateBatch` 和 `RuntimeScalarExpressionNode::evaluateBatch` 中，每次求值（积分点的紧密循环）都会调用 `std::vector<double> lhsValues(dest.size());`。在 FEM 组装的底层紧密循环中动态分配堆内存，会导致灾难性的性能瓶颈（Cache Miss 和 Allocator 锁竞争）。
3. **反模式：推模型（Push-Model）的属性注入**
   * **问题**：`PhysicsProblemBuilder` 负责去材质库里查找特定名称（如 `kPropThermalConductivity`），然后显式调用 `solver->setThermalConductivity(...)`。这导致 Builder 必须知道每一个求解器的内部实现细节，后续扩展新的物理场时必须修改 Builder。
4. **冗余：材质系统与变量系统的两层皮**
   * **问题**：`MaterialDatabase` 存储了一遍字符串，然后 `ProblemBuilder` 又把这些字符串取出来编译成 `VariableNode`，存在两层状态同步问题。

---

### 破坏式重构步骤计划

为了实现一个 **简洁、高效、一致的纯 DAG+表达式系统**，建议按照以下 5 个阶段进行“破坏式重构”。每一步都可独立验证编译。

#### 第一步：消除硬编码，实现“动态符号与场”注册中心 (Dynamic Registry)
**目标**：彻底干掉 `classifyRuntimeField`、`BuiltInSymbolKind` 等带有硬编码 "x, y, T, V" 的枚举。

1. **修改 `VariableManager`**：提供通用的运行时变量注册接口，而不是依赖外部传入的回调工厂 `GraphExternalSymbolBinder`。
2. **重构**：允许将任何物理场（`GridFunction`）或内置坐标直接注册为节点。

```cpp
// 1. 新增接口：提供场求值的基类
class FieldProvider {
public:
    virtual ~FieldProvider() = default;
    virtual double evaluate(const EvaluationContext& ctx, size_t pointIndex) const = 0;
    // 如果支持梯度
    virtual Vector3 evaluateGradient(const EvaluationContext& ctx, size_t pointIndex) const = 0;
};

// 2. VariableManager 提供直接注册物理场的接口
class VariableManager {
public:
    // ...
    void registerField(std::string name, std::shared_ptr<FieldProvider> field);
    void registerCoordinate(std::string name, int axis); // x=0, y=1, z=2
    void registerTime(std::string name);
};
```
* **验证机制**：删除 `classifyRuntimeField` 和 `classifyBuiltInSymbol`。测试求解器是否能通过名称 `"T"` 正常获取温度场的值。

#### 第二步：消灭 C++ 物理耦合节点，升级表达式解析器 (Parser Upgrade)
**目标**：彻底删除 `ProductScalarNode`、`JouleHeatNode` 和 `ThermalExpansionStressNode`。让解析器承担真正的 DAG 计算责任。

1. **升级 `ExpressionParser`**：使你的数学解析库支持矩阵运算和微分算子（如 `grad(V)`, `dot(a, b)`, `trace(A)`, `sym(A)`）。
2. **用表达式重写耦合逻辑**：将数百行的 C++ Coded Node 替换为一行注册代码。

```cpp
// 以前的做法 (physics_problem_builder.cpp):
// const VariableNode* joule = problem.ownNode(std::make_unique<JouleHeatNode>(V_field, sigmaByDomain));

// 重构后的做法：完全基于 DAG 和表达式
// 只需在问题初始化时注册焦耳热的数学表达式
problem.varManager.registerScalarExpression(
    "JouleHeat", 
    "dot(grad(V), sigma * grad(V))"
);
problem.heatTransfer->setHeatSource(activeDomains, problem.varManager.get("JouleHeat"));

// 热膨胀应力同理：
problem.varManager.registerMatrixExpression(
    "ThermalStrain",
    "alpha * (T - T_ref)"
);
problem.varManager.registerMatrixExpression(
    "ThermalStress",
    "2 * mu * sym(ThermalStrain) + lambda * trace(ThermalStrain) * Identity"
);
```
* **验证机制**：删掉所有的 C++ 自定义算子类，代码行数大幅减少。多物理场耦合仅仅体现为往 DAG 中添加公式图节点。

#### 第三步：重构求值热路径，实现零动态内存分配 (Zero-Allocation Hot-Path)
**目标**：在 `evaluateBatch` 中杜绝所有的 `std::vector` 动态分配。

1. **引入 `EvaluationWorkspace` / 提前预分配**：DAG 节点求值时，临时结果应该存放在预先分配的线性内存池（或线程局部缓存）中。
2. **重写 Batch 签名**：

```cpp
// 传递一个可复用的上下文工作区，避免每次 new memory
struct EvaluationWorkspace {
    std::vector<double> scratchpad; 
    size_t offset = 0;
    
    std::span<double> allocate(size_t size) {
        // 从 scratchpad 中快速划出一块内存（仅移动指针）
    }
    void reset() { offset = 0; }
};

// VariableNode 签名修改
virtual void evaluateBatch(const EvaluationContext& ctx, 
                           EvaluationWorkspace& workspace, 
                           std::span<double> dest) const = 0;
```
* **验证机制**：在 `evaluateBatch` 内部使用工具（如 Valgrind 或重载 global `new`）断言：在执行装配（`assemble()`）期间，内存分配次数必须为 0。

#### 第四步：控制反转，求解器采用“拉取式”依赖 (Pull-based Dependency)
**目标**：消除 `PhysicsProblemBuilder` 作为 "God Class" 的设计，将它变成纯粹的工厂，让各个 Solver 自己去 DAG 中寻找自己需要的参数。

1. **解除 Builder 的负担**：Builder 不需要知道 `HeatTransferSolver` 需要 `rho` 和 `Cp`。
2. **求解器自注册和拉取**：

```cpp
void HeatTransferSolver::initializeDependencies(const VariableManager& varMgr) {
    // 求解器自己定义自己依赖的变量名称约定
    this->k_node = varMgr.get("thermalconductivity");
    this->rho_node = varMgr.get("density");
    this->cp_node = varMgr.get("heatcapacity");
    
    if (!k_node) {
        MPFEM_THROW("HeatTransfer requires 'thermalconductivity' to be defined in the expression graph.");
    }
}
```
* **验证机制**：删除 `PhysicsProblemBuilder` 中用于查找并调用 `setThermalConductivity` 等属性赋值的长串代码。Builder 只需要无脑解析 XML，把所有表达式丢进 `VariableManager`，然后让求解器自行校验 DAG。

#### 第五步：合并材质系统与 DAG (Domain-Aware DAG)
**目标**：处理按物理区域（Domain ID）变化的参数，废弃两层抽象的 `MaterialDatabase`。

1. **区域条件表达式 (Piecewise / Select)**：对于多个区域具有不同材料参数的问题，应该在 DAG 中体现为一个**分支节点**或原生的条件表达式。
2. **实现**：

可以在表达式语法中支持 `domain` 关键字或 `select` 函数：
```text
density = select(domain == 1: 7800, domain == 2: 1000)
```
或者在 C++ `VariableManager` 内部实现一个 `PiecewiseNode`，专门根据 `ctx.domainId` 路由到不同的子表达式节点。
* **验证机制**：彻底删除 `MaterialDatabase` 类（它只需要简化为 XML 解析的一个过程），所有的材料属性最终都是 `VariableManager` 中的一个带 Domain 路由的节点。

### 总结
执行以上 5 步后，你的架构将变成纯粹的数据流：
`XML输入 -> 表达式注册到 VariableManager -> DAG 图编译 (拓扑排序) -> 求解器直接从 DAG 取用变量 (零内存分配) -> 并行组装 -> 求解`。

这不仅在代码量上能精简 40% 以上的样板逻辑，更将极大提升架构应对新增物理场及方程体系的通用性。
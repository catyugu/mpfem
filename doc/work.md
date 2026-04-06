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

### 一、 核心反模式与设计缺陷诊断

1. **反模式：绕过表达式图手写AST（局部且冗余）**
   * **位置**：`src/physics/heat_transfer_solver.cpp` 中的 `ProductScalarNode`。
   * **问题**：这是一个巨大的反模式。系统明明已经有 `ExpressionParser` 和 `VariableManager`，求解器却为了计算 $\rho \cdot C_p$ 在自己的 CPP 文件里硬编码写了一个乘法 AST 节点。这导致表达式解析器形同虚设，矩阵和张量的运算在未来也需要手动写 `DotProductNode`, `CrossProductNode`，代码将无限膨胀。
2. **冗长：物理场参数绑定硬编码**
   * **位置**：`HeatTransferSolver` 的 `setThermalConductivity`, `setHeatSource` 等方法。
   * **问题**：如果增加一个物理现象（比如热辐射），就要给求解器加一个 `std::vector<RadiationBinding>` 和对应的 `set` 方法。这使得求解器代码又长又死板。
3. **冗长：`FieldValues` 违反单一职责与零经验法则 (Rule of Zero)**
   * **位置**：`FieldValues` 的拷贝构造函数和赋值运算符。
   * **问题**：内部混杂了“场存储”、“时间步进”和“环形缓冲区管理”。手动编写深拷贝逻辑不仅容易出错，而且极度冗长。
4. **割裂：`FieldValues`（数据）与 `VariableManager`（图节点）脱节**
   * **问题**：创建物理场时没有自动注册到 DAG，导致计算图在求值时需要外部代码显式把场数据“喂”给变量系统，存在潜在的数据不一致和依赖循环（解析表达式需要场，更新场需要计算表达式）。

---

### 二、 分步重构指南（可执行、可验证）

为了实现“完全用自动DAG+表达式解析的变量系统”，建议按照以下步骤进行破坏式重构：

#### Step 1: 消除硬编码 AST，强制统一使用 VariableManager
**目标**：砍掉求解器中所有的自制 AST 节点，完全依赖表达式引擎。

1. **操作**：删除 `heat_transfer_solver.cpp` 中的 `ProductScalarNode` 类。
2. **操作**：修改 `HeatTransferSolver::setMassProperties` 接口。不再接受两个节点（$\rho$ 和 $C_p$），而是接受单一的表达式结果节点。
3. **调用侧重构**：用户的装配逻辑应当变为：
   ```cpp
   // 重构前：
   solver.setMassProperties(domain, varMgr.get("rho"), varMgr.get("Cp"));
   
   // 重构后：利用 VariableManager 的自动 DAG 解析
   varMgr.registerExpression("ThermalMass", "rho * Cp"); 
   solver.setMassProperties(domain, varMgr.get("ThermalMass"));
   ```
4. **验证**：编译 `heat_transfer_solver.cpp`，确认单元测试中热传导质量矩阵依然能够正确组装。

#### Step 2: 重构 FieldValues，实现状态与历史的解耦 (Rule of Zero)
**目标**：干掉 `FieldValues` 中成百上行的手动深拷贝代码，使其成为纯粹的数据容器。

1. **操作**：提取出一个 `FieldState` 类来管理单个场及其历史，利用 `GridFunction` 的拷贝构造来实现深拷贝（如果 `GridFunction` 没有，请为其添加 default copy 语义）。
   ```cpp
   // 重构后的 field_values.hpp
   struct FieldState {
       GridFunction current;
       std::vector<GridFunction> history; // 放弃 unique_ptr，直接存对象，利用 vector 的值语义
       int historyHead = 0;
       int maxHistory = 0;
       bool isVector = false;
       int vdim = 1;

       // 默认拷贝构造即可完成深拷贝！无需手写！
   };

   class FieldValues {
   public:
       // 纯粹的 unordered_map，自动获得 default copy constructor，彻底消灭手写的大段拷贝代码！
       FieldValues() = default; 
   private:
       std::unordered_map<std::string, FieldState> fields_;
       int maxHistorySteps_ = 0;
   };
   ```
2. **验证**：检查所有引用 `FieldValues` 拷贝的地方，确保内存不泄露，代码行数应能减少至少 50 行以上。

#### Step 3: 将 FieldValues 深度绑定到 VariableManager (自动注册)
**目标**：实现“只要物理场被创建，表达式中就可以直接写它的名字”。

1. **操作**：修改 `FieldValues::createScalarField` 和 `createVectorField`，让它们持有一个指向 `VariableManager` 的引用，并在创建场后自动调用 `varMgr.registerGridFunction`。
   ```cpp
   void createScalarField(std::string_view id, const FESpace* fes, VariableManager& varMgr, Real initVal = 0.0) {
       auto& entry = fields_[std::string(id)];
       entry.current = GridFunction(fes, initVal); // 基于 Step 2 的值语义
       // 自动注册进 DAG，表达式可以直接识别 "Temperature"
       varMgr.registerGridFunction(std::string(id), &entry.current); 
   }
   ```
2. **验证**：编写测试：创建一个名为 `V` 的场，直接在 `VariableManager` 注册表达式 `"V^2 + 1"`，验证是否能正确计算而无需额外绑定。

#### Step 4: 统一求解器的参数绑定接口 (数据驱动配置)
**目标**：消除求解器中泛滥的 `setXXX` 方法，改为基于统一标识符的参数字典。所有参数均视为 `VariableNode`。

1. **操作**：在 `PhysicsFieldSolver` 基类中引入一个通用的参数映射：
   ```cpp
   // physics_field_solver.hpp
   class PhysicsFieldSolver {
   public:
       // 统一接受 domains/boundaries 和 VariableNode
       void setParameter(const std::string& paramName, const std::set<int>& regions, const VariableNode* node) {
           parameters_[paramName].push_back({regions, node});
       }
   protected:
       struct ParamBinding { std::set<int> regions; const VariableNode* node; };
       std::unordered_map<std::string, std::vector<ParamBinding>> parameters_;
   };
   ```
2. **操作**：在 `HeatTransferSolver` 中，彻底移除 `setThermalConductivity` 等方法。组装时直接从字典取值：
   ```cpp
   // heat_transfer_solver.cpp -> assemble()
   auto conductivities = parameters_["Conductivity"];
   for (const auto& binding : conductivities) {
       matAsm_->addDomainIntegrator(
           std::make_unique<DiffusionIntegrator>(binding.node), 
           binding.regions
       );
   }
   ```
3. **收益**：这使得系统具备了极高的扩展性。你的 XML/JSON 配置文件可以直接将标签名映射为这里的 `paramName`，中间不需要写任何 C++ 胶水代码。

#### Step 5: 升级表达式后端，支持标量/向量/矩阵多态
**目标**：彻底打通标量、向量、矩阵计算（您的最终意图）。

1. **操作**：修改 `EvaluationContext` 和 `VariableNode::evaluateBatch`，强制 `dest` 不再是展平的 `std::span<double>`，而是带有步长（Stride）或直接使用支持多维的张量视图（如基于 `std::mdspan` 或 Eigen 的 `Map`）。
   ```cpp
   class VariableNode {
   public:
       // ctx 传递评估上下文，dest 使用 mdspan 或扁平连续内存结合 shape() 信息
       virtual void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const = 0;
   };
   ```
2. **操作**：`ExpressionParser::ExpressionProgram` 需要增强类型推导：在 DAG 的编译期 (`compileGraph`) 执行类型推导检查（例如：检查“温度(标量) * 位移(向量)”的结果是否被正确推导为向量节点）。
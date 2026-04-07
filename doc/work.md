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

### 🚨 识别出的核心反模式 (Anti-Patterns)

1. **伪DAG与硬编码物理耦合 (Pseudo-DAG & Hardcoded Physics)**
   * **问题**: `JouleHeatNode` 和 `ThermalExpansionStressNode` 是完全用C++硬编码的变量节点。在真正的DAG表达式系统中，焦耳热应该仅仅是一个表达式 `sigma * grad(V) * grad(V)`，热应力应该仅仅是 `2*mu*sym(grad(u)) + lambda*tr(sym(grad(u)))*I - (3*lambda+2*mu)*alpha*(T-Tref)`。目前通过硬编码C++类去手动拉取场变量（`heat_->field().eval`），完全绕过了你设计的自动表达式解析系统。
2. **割裂的张量表达式后端 (Fragmented Tensor Backend)**
   * **问题**: 在 `expression_parser.cpp` 中，矩阵表达式被解析为 9 个独立的标量 `ExpressionProgram` (`components`)。这意味着 VM 根本不懂矩阵运算，它只是在跑 9 次标量 VM。这无法支持真正的张量代数（如点乘、叉乘、矩阵乘法），也无法支持向量。
3. **VM 求值层的性能灾难 (Performance Catastrophe in VM)**
   * **问题**: `evaluate_single_vm` 在每次被调用时（即每个积分点），都会 `std::vector<double> stack; stack.reserve(16);`。在有限元组装中，这意味着数百万次的堆内存分配。
4. **通过字符串拼接管理域属性 (String-Typing for Domains)**
   * **问题**: `PhysicsProblemBuilder` 中充斥着 `E_1`, `E_2` 这样的字符串拼接来区分不同 Domain 的材质属性。这种设计非常脆弱，容易导致拼写错误，且无法利用类型系统或图论验证依赖。

---

### 🛠️ 步骤化重构方案 (Step-by-Step Refactoring Plan)

你的目标是**完全统一到“表达式+DAG”后端**。以下是破坏性重构的步骤，每一步都保证系统更接近最终目标。

#### 第一步：重构虚拟机与 AST，实现原生张量支持 (Native Tensor VM)

目前你的表达式引擎是标量的。我们需要让底层的 `ExprValue` 和 VM 栈直接支持标量、向量和矩阵。

**操作步骤：**
1. **统一数据结构**：修改 VM 栈，使其不再是 `double`，而是一个统一的、无堆分配的定长结构体（如 `union` 或 `std::array<double, 9>`，加上类型枚举）。
2. **扩展 OpCode**：添加张量操作指令，如 `MatMul` (矩阵乘法), `Dot` (点乘), `Grad` (求梯度)。
3. **消除 Components 拆分**：删除 `MatrixTemplate` 拆分为 9 个子程序的逻辑。让 AST 直接解析形如 `[a, b, c; d, e, f; g, h, i]` 的语法，生成一个统一的 `AstNode::MatrixLiteral`，最终编译出直接操作矩阵的 OpCode。
4. 但是COMSOL的材料属性配置文件中使用的仍然是形如：
```xml
<set name="electricconductivity" value="{'1/(1.72e-8*(1+0.0039*(T-298)))[S/m]','0','0','0','1/(1.72e-8*(1+0.0039*(T-298)))[S/m]','0','0','0','1/(1.72e-8*(1+0.0039*(T-298)))[S/m]'}"/>
<set name="thermalexpansioncoefficient" value="{'17e-6[1/K]','0','0','0','17e-6[1/K]','0','0','0','17e-6[1/K]'}" />
```
你需要兼容一下这个格式。

**验证点**：编译并运行一个测试，计算 `[1,0,0; 0,1,0; 0,0,1] * [x, y, z]^T`，确保 VM 能够原生输出向量。

#### 第二步：统一 VariableNode (Unify Variable Nodes)

消除 `RuntimeScalarExpressionNode` 和 `RuntimeMatrixExpressionNode` 的硬编码区分，所有的表达式节点应该统一为一个 `RuntimeExpressionNode`。

**操作步骤：**
1. 合并节点：
   ```cpp
   // 重构后的统一表达式节点
   class RuntimeExpressionNode final : public VariableNode {
   public:
       RuntimeExpressionNode(ExpressionProgram program, std::vector<const VariableNode*> deps);
       TensorShape shape() const override { return program_.shape(); }
       void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override;
   private:
       ExpressionProgram program_;
       std::vector<const VariableNode*> dependencies_;
       // ...
   };
   ```
2. **批量执行 VM**：将 `evaluate_single_vm` 改造为 `evaluate_batch_vm`。在最外层分配一次栈（根据最大深度预分配），然后对一个 batch 的所有积分点执行一个紧凑的循环，彻底消除 `std::vector` 的循环内分配。

**验证点**：运行标量和矩阵表达式，确保输出正确，且性能得到数量级的提升。

#### 第三步：引入空间算子并抹除硬编码物理节点 (Eliminate Hardcoded Physics via Operators)

这是实现自动 DAG 的核心。我们不能在 C++ 中硬写 `JouleHeatNode`。我们必须让 `GridFunctionNode` 提供它的梯度，并让表达式解析器支持 `grad(V)`。

**操作步骤：**
1. 扩展 `GridFunctionNode`，使其支持除了值 (Value) 之外的算子求值 (如 Gradient)。
   ```cpp
   // 在表达式语法中支持 `grad(V)` 或 `del(V)`
   // 解析器将其识别为一个特殊函数，并向 VariableManager 请求 V 的梯度节点
   ```
2. 在 `PhysicsProblemBuilder.cpp` 中**彻底删除 `JouleHeatNode` 和 `ThermalExpansionStressNode`**。
3. 替换为纯表达式注册：
   ```cpp
   // 焦耳热就是一句纯纯的表达式
   problem.globalVariables_.registerExpression("JouleHeat", "dot(grad(V), sigma * grad(V))");
   problem.heatTransfer->setHeatSource(activeDomains, problem.globalVariables_.get("JouleHeat"));
   
   // 热应变/应力同样纯表达式化
   problem.globalVariables_.registerExpression("ThermalStrain", "alpha * (T - T_ref)");
   ```

**验证点**：编译通过后，焦耳热和热膨胀的测试结果与重构前完全一致，但代码减少了数百行复杂的 C++。

#### 第四步：重构 Material Domain 注册 (Refactor Domain Binding)

放弃使用 `"E_1"`, `"E_2"` 这种基于字符串拼接的域隔离，改用 `DomainSelectorNode`（域选择器节点），由 DAG 在求值时根据 `ctx.domainId` 自动路由。

**操作步骤：**
1. 引入一种新的 DAG 节点：多路复用节点 (Multiplexer Node)。
   ```cpp
   class DomainMultiplexerNode final : public VariableNode {
   public:
       // map: domainId -> sub-node
       void addDomain(int domainId, const VariableNode* node);
       void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override {
           // 根据 ctx.domainId 转发给对应的子节点
           children_.at(ctx.domainId)->evaluateBatch(ctx, dest);
       }
   };
   ```
2. 修改 `PhysicsProblemBuilder`：
   ```cpp
   // 用户代码中，材料属性依然叫 "E", "nu", "sigma"
   auto E_node = std::make_unique<DomainMultiplexerNode>();
   for (int domId : domains) {
       E_node->addDomain(domId, parser.compile(mat_expr[domId]));
   }
   problem.globalVariables_.adoptNode("E", std::move(E_node));
   ```
3. 这样在热应力表达式中，直接写 `"alpha * (T - T_ref)"` 即可，底层 `alpha` 节点在执行时会自动根据当前积分点所在的 `domainId` 取对应材料的膨胀系数。

**验证点**：多材质模型可以正常求解，且 `VariableManager` 中不再有杂乱的带后缀变量名。
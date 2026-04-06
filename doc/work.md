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

### 🛠️ 步骤化破坏式重构方案

每一步都保证系统处于可编译、可运行的完整状态，宁可采用破坏性修改也不保留向后兼容的包袱。

#### 统一数据载体，消灭标量与矩阵的类型分裂 (Unify Tensor Backend)

**目标**: 将 `double` 和 `Matrix3` 统一为泛型的张量变体，合并所有 `compileX` 方法。

1.  **引入 `Value` 变体类型**: 在 `core/types.hpp` 中引入统一的表达式值类型：
    ```cpp
    // src/core/types.hpp
    #include <variant>
    namespace mpfem {
        using ExprValue = std::variant<double, Vector3, Matrix3>;
        // 辅助函数用于获取类型和维度
    }
    ```
2.  **合并 Program**: 在 `expression_parser.hpp` 中，删除 `ScalarProgram` 和 `MatrixProgram`，统一为一个 `ExpressionProgram`。
    ```cpp
    class ExpressionProgram {
    public:
        VariableShape shape() const; 
        ExprValue evaluate(std::span<const ExprValue> values) const;
    };
    // compileScalar 和 compileMatrix 合并为：
    ExpressionProgram compile(const std::string& expression) const;
    ```
3.  **合并 Node**: 在 `variable_graph.hpp` 中，删除 `ConstantScalarNode`, `RuntimeScalarExpressionNode`, `RuntimeMatrixExpressionNode`。只保留一个通用的 `ExpressionNode`。
    * **验证标准**: 此时所有的测试用例应该能通过新的单一接口 `compile("...")` 运行，无需外部指定它是标量还是矩阵。

#### 净化 DAG，移除内置符号硬编码 (Purify DAG Leaves)

**目标**: 移除 `BuiltInSymbolKind`，将物理场和时空坐标转变为标准的 DAG 节点。

1.  **移除硬编码**: 删除 `variable_graph.cpp` 中的 `BuiltInSymbolKind` 及其 `switch` 逻辑。
2.  **实现专用叶子节点**:
    ```cpp
    class SpatialCoordinateNode : public VariableNode {
        // shape = Vector3, evaluateBatch 时直接从 ctx.physicalPoints 拷贝
    };
    class TimeNode : public VariableNode {
        // shape = Scalar, evaluateBatch 时返回 ctx.time
    };
    class FieldEvaluationNode : public VariableNode {
        // 专门处理其他物理场（如 T, V）的求值
    };
    ```
3.  **预注册**: 修改 `VariableManager` 的构造函数，默认自动注册 `x`, `y`, `z`, `t` 作为内部保留节点。例如，注册 `x` 的表达式为提取 `SpatialCoordinateNode` 的第 0 个分量。
    * **验证标准**: 表达式 `"x + t * 2"` 仍然可以求值，但是调度不再经过任何 `if(symbol == "t")` 分支，而是纯粹的 DAG 依赖传递。

#### 增强解析器，实现公式化的多物理场耦合 (Formula-based Coupling)

**目标**: 淘汰 `JouleHeatNode` 和 `ThermalExpansionStressNode`，用原生表达式取代。

1.  **扩展 AST 支持算子**: 在 `expression_parser.cpp` 中支持空间微分算子和张量算子，例如 `grad(field)`, `dot(A, B)`, `sym(A)`。
2.  **重写耦合逻辑**: 在构建耦合时，直接向 `VariableManager` 注册字符串表达式。
    * **重构前**:
        ```cpp
        // src/problem/physics_problem_builder.cpp
        const VariableNode* joule = problem.ownNode(
            std::make_unique<JouleHeatNode>(V_field, sigmaByDomain));
        ```
    * **重构后 (破坏式)**:
        ```cpp
        // 直接编译公式，解析器负责推导 grad(V) 并在执行时通过 FieldEvaluationNode 获取梯度
        problem.materials.registerExpression("Q_joule", "dot(grad(V), sigma * grad(V))");
        ```
3.  **删除旧类**: 彻底删除 `physics_problem_builder.cpp` 中的 `JouleHeatNode` 和 `ThermalExpansionStressNode`。
    * **验证标准**: 稳态或瞬态的焦耳热计算结果与旧版本完全一致，但代码量锐减。

#### 引入场自注册机制，废弃硬编码解析器
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

#### 统一全局 DAG，消灭临时管理器与域缓存
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

#### 用动态解析替换局部硬编码 AST (消灭 `ProductScalarNode`)
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

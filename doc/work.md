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

每一步都保证系统处于可编译、可运行的完整状态，宁可采用破坏性修改也不保留向后兼容的包袱。

### 引入场自注册机制，废弃硬编码解析器
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

### 统一全局 DAG，消灭临时管理器与域缓存
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

### 用动态解析替换局部硬编码 AST (消灭 `ProductScalarNode`)
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

### 重构表达式后端（从 AST 树到线性化指令流）
**目标**：消除递归求值，引入基于栈（Stack-based VM）或扁平数组的指令流，极大提升单点求值性能。

1.  **定义操作码（OpCode）和指令：**
    ```cpp
    enum class OpCode : uint8_t { Constant, LoadVar, Add, Sub, Mul, Div, Pow, Sin, Cos /*...*/ };
    struct Instruction {
        OpCode op;
        double value; // 用于 Constant
        int index;    // 用于 LoadVar
    };
    ```
2.  **修改编译器：**
    在 `ScalarAstCompiler::compile()` 后增加一个步骤（或者直接在解析时生成），将 AST 遍历一遍，转换为 `std::vector<Instruction>`（后缀表达式形式）。
3.  **重写 Evaluate 函数：**
    ```cpp
    // 替换掉原来的 evalAstNode 递归调用
    double ExpressionProgram::evaluate_single(std::span<const double> vars) const {
        std::vector<double> stack; // 实际实现中可用固定大小的 std::array 或小端优化以避免堆分配
        for (const auto& inst : instructions_) {
            switch (inst.op) {
                case OpCode::Constant: stack.push_back(inst.value); break;
                case OpCode::LoadVar: stack.push_back(vars[inst.index]); break;
                case OpCode::Add: {
                    double b = stack.back(); stack.pop_back();
                    stack.back() += b; 
                    break;
                }
                // ... 其他操作
            }
        }
        return stack.back() * multiplier_;
    }
    ```
*验证*：原有的所有标量表达式单元测试应该无缝通过，且性能提升。

### 泛化变量形态（统一标量、向量与矩阵）
**目标**：删除 `RuntimeScalarExpressionNode` 和 `RuntimeMatrixExpressionNode`，用单一类处理任意维度（Shape）。

1.  **统一 VariableShape 和 Dimensions：**
    ```cpp
    // 废弃 enum class VariableShape，改用动态或固定的 Shape 结构
    struct TensorShape {
        std::vector<int> dims; // [] 为标量, [3] 为向量, [3,3] 为矩阵
        size_t flatSize() const; // 返回总组件数
    };
    ```
2.  **合并 Node 类为统一的 `ExpressionNode`：**
    让 `ExpressionNode` 内部持有一个 `std::vector<ExpressionProgram>`，对应张量的每一个展平（Flattened）的分量。
    ```cpp
    class ExpressionNode final : public VariableNode {
    public:
        // 无论是标量(1个程序), 向量(3个程序), 还是矩阵(9个程序)，都用统一逻辑
        void evaluateBatch(...) override {
             // 逻辑简化为遍历 programs 数组并写入对应的内存偏移
        }
    private:
        TensorShape shape_;
        std::vector<ExpressionProgram> componentPrograms_; 
    };
    ```
*验证*：替换原有 API 后，标量和矩阵的行为在外部表现一致，同时天然获得了对向量的支持。

### 真正激活 DAG（全局工作区模型）
**目标**：消除 `evaluateDependencyBlocks` 的局部递归，启用全局的 `executionPlan_`。

1.  **引入 VariableWorkspace（求值工作区）：**
    在图开始评估前，分配一块连续内存，用于存放当前所有节点的输出。
    ```cpp
    struct VariableWorkspace {
        // key 为 Node 指针，value 为该 Node 对应所有点的计算结果数组
        std::unordered_map<const VariableNode*, std::vector<double>> buffers;
    };
    ```
2.  **重写 VariableManager 的求值逻辑：**
    由 Manager 负责驱动（而不是 Node 驱动）。
    ```cpp
    void VariableManager::evaluateGraph(const EvaluationContext& ctx, VariableWorkspace& workspace) {
        // 必须按拓扑排序执行！
        for (const VariableNode* node : executionPlan_) {
            auto& outputBuffer = workspace.buffers[node];
            outputBuffer.resize(ctx.physicalPoints.size() * node->shape().flatSize());
            
            // 节点不再负责去 resolve 自己的依赖，它的依赖肯定已经在这个循环前面计算好并存在 workspace 中了！
            node->evaluateBatch(ctx, workspace, outputBuffer); 
        }
    }
    ```
3.  **精简 Node 的求值：**
    `ExpressionNode::evaluateBatch` 现在只需要直接从 `workspace.buffers[dependencyNode]` 中读取数据作为输入，不再调用任何子节点的 evaluate。
*验证*：打印求值日志，验证每个节点严格只被求值一次，没有重复计算。
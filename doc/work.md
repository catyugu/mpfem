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
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

通过对提供的项目源码分析，该有限元（FEM）框架的整体结构清晰，但在具体实现上存在明显的**设计反模式（Anti-patterns）、冗长代码（Verbosity）、多余的向后兼容接口**以及**性能/内存瓶颈**。

因为不需要考虑向后兼容，我们可以放开手脚进行现代 C++ 风格的深度重构。以下是针对该代码库的步骤化重构方案。

---

### 1. 核心问题诊断与反模式识别

1. **宏污染与核心逻辑强耦合（Anti-pattern / Verbosity）**：`assembler.cpp` 中为了实现 OpenMP 并行，充斥着大量的 `#ifdef _OPENMP` 分支，导致真正的组装逻辑（数学积分）被淹没在线程缓冲区初始化和锁中。
2. **AST 树的指针嵌套导致内存碎片与缓存不友好（Performance/Memory）**：在 `expression_parser.cpp` 中，AST 节点包含 `std::vector<std::unique_ptr<AstNode>> args`。对每一求值点执行递归指针调用会导致极差的 CPU Cache 命中率和多余的堆内存分配。
3. **巨型 Switch-Case 代码重复（Verbosity）**：`geometry_mapping.cpp` 和 `h1.cpp` 中包含两套冗长、硬编码的形状函数与形函数的导数表达式。
4. **向后兼容的妥协（Redundancy）**：`Tensor` 类存在大量的 `asVector3`、`matrix3` 等为了兼容旧版代码而存在的转换函数，破坏了封装的统一性。

---

### 2. 步骤化重构方案

#### 步骤 1：清理底层计算内核与精简数据结构（删除向后兼容）
**目标**：消除 `core/kernels.hpp`，精简 `Tensor` 和 `VariableGraph`，统一采用 `Eigen` 内置的定长矩阵优化。
。
* **精简 `Tensor` 类**：删除所有兼容性构造和检查。直接保留 `TensorShape` 和 `TensorData` 即可，让调用方直接通过标准的数学运算符操作，不要保留 `asMatrix3()`, `asVector3()` 等冗余方法，只给维度无关的 `vector()`, `matrix()` 接口，强制接口统一。

#### 步骤 2：消除 OpenMP 宏污染，提取并行装配模式
**目标**：将 `assembler.cpp` 的 300 多行代码缩减 50% 以上，分离“线程调度”与“数学装配”。

* **重构动作**：在 `core` 模块中实现一个泛型的 `parallel_for` 迭代器，将复杂的 OpenMP 分支收敛到一处。

```cpp
// 重构后的装配循环 (assembler.cpp)
void BilinearFormAssembler::assemble() {
    mat_.setZero();
    triplets_.clear();

    // 将繁杂的 #ifdef 抽象为一个并行循环组件
    utils::ParallelFor<ThreadBuffer, std::vector<SparseMatrix::Triplet>>(
        mesh->numElements(),
        [&](ThreadBuffer& buf, int e, auto& localTriplets) {
            // ... 纯粹的元素装配逻辑 ...
            buf.elmatVector.setZero();
            for(auto& integ : domainIntegs) {
               integ->assembleElementMatrix(ref, trans, buf.dynMatrix);
               buf.elmatVector += buf.dynMatrix;
            }
            // 写入 localTriplets
        },
        [&](auto& threadLocalTriplets) {
            // reduce 操作：合并所有的 localTriplets
            triplets_.insert(triplets_.end(), threadLocalTriplets.begin(), threadLocalTriplets.end());
        }
    );
    mat_.setFromTriplets(std::move(triplets_));
}
```

#### 步骤 3：消除 Integrator 中的 Boilerplate 代码
**目标**：解决 `integrators.cpp` 中每个子类都重复写高斯积分点遍历的问题。

* **重构动作**：在基类或者工具命名空间中提供一个高阶函数 `integrateElement`。

**重构前：** 每个 Integrator 都要写 `for (int q = 0; q < nq; ++q) { trans.setIntegrationPoint(...); Real w = ...; }`
**重构后：**
```cpp
void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& ref, ElementTransform& trans, Matrix& elmat) const {
    elmat.setZero(nd, nd);
    // 高阶函数抽象遍历与权重计算
    quadrature::integrateElement(ref, trans, [&](int q, Real weight) {
        const Matrix3 D = evalMatrixNode(coef_, trans);
        // 直接执行核心数学逻辑，无需再关心 trans 和 integration point
        const Vector3 physGrad = trans.transformGradient(...);
        elmat.noalias() += weight * (gradMat * D * gradMat.transpose());
    });
}
```

#### 步骤 4： AST 求值引擎的“扁平化” / 引入轻量级虚拟机 (VM)
**目标**：彻底解决 `expression_parser.cpp` 中庞大、低效的树形指针结构。这能极大节省内存，避免递归函数调用，并且加快大批量数据的求值效率。

* **重构动作**：将 `AstNode` 的树结构（Tree）在编译后（`VariableManager::compile`）展平为线性的字节码（Bytecode）数组（即后缀表达式/逆波兰表示）。

```cpp
// 线性指令结构，替代原有的指针树结构
struct Instruction {
    OpCode op;
    int var_index = -1;
    Real constant_val = 0.0;
};

class CompiledExpressionNode final : public VariableNode {
    std::vector<Instruction> bytecode_; // 扁平连续内存，Cache极度友好

    void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override {
        // 使用一个固定大小的栈进行极速的循环求值
        std::vector<Real> stack;
        stack.reserve(16); 
        
        for (size_t i = 0; i < dest.size(); ++i) {
            for (const auto& instr : bytecode_) {
                switch (instr.op) {
                    case OpCode::PushConst: stack.push_back(instr.constant_val); break;
                    case OpCode::PushVar: stack.push_back(fetchVar(instr.var_index)); break;
                    case OpCode::Add: {
                        Real b = stack.back(); stack.pop_back();
                        stack.back() += b;
                        break;
                    }
                    // ... 无递归，全是内存连续的 O(N) 循环
                }
            }
            dest[i] = Tensor::scalar(stack.back());
            stack.clear();
        }
    }
};
```

#### 步骤 5：消除重叠功能，统一 Shape 映射
**目标**：解决 `geometry_mapping.cpp`（完全手动写死的多项式）与 `h1.cpp` 中（拉格朗日多项式引擎）逻辑重合且代码冗长的问题。

* **重构动作**：废弃 `GeometryMapping` 里 500 多行的巨型 Switch 语句。改用基于张量积（Tensor Product）的统一生成器。
* 既然项目在 `h1.cpp` 定义了 `Lagrange1D` 和 `Barycentric`，直接让 `ReferenceElement` 依赖 `FiniteElement::evalShape` 即可，**只保留唯一的真实数据源**，将几何映射和场函数插值合并。所有的 `evalDerivatives` 可以通过自动生成 1D 导数后依据链式法则做乘积，而不是每个三维体硬写解析导数，让代码从近千行缩短至不足百行。

### 总结
通过上述重构，您可以剔除向后兼容和低效优化的历史包袱，代码不仅能在接口上形成绝对统一（一切表达式皆编译为 Bytecode，一切多线程皆封装为 ParallelFor，一切积分皆高阶抽象），而且在堆内存分配和 CPU Cache 预取上获得极大的性能提升。
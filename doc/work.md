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

通过对代码的分析，现有的实现存在以下严重的设计反模式和耦合问题：

### 🚨 识别出的设计反模式与问题
1. **胖接口与破坏封装 (Broken Encapsulation)**：`HeatTransferSolver` 为了适应时间积分器，被迫暴露出内部状态缓存 `stiffnessMatrixBeforeBC_` 和 `rhsBeforeBC_`。不仅打破了封装，还导致 `StructuralSolver` 和 `HeatTransferSolver` 的 API 极度不一致。
2. **急切计算导致冗余 (Eager Computation)**：即使是在 `SteadyProblem`（稳态求解）中，只要材料定了密度和比热，`HeatTransferSolver::initialize` 就会去强行组装一个无用的 `massMatrix`。
3. **极高的模块循环依赖 (Cyclic Dependency)**：`time/bdf1_integrator.cpp` 中竟然包含了 `problem/transient_problem.hpp` 并硬编码 `if (!problem.heatTransfer) return false;`。时间积分算法不应该知道“热传导”的存在，它只能看到 $M$、$K$ 矩阵，这种硬编码导致代码无法复用于力学或电磁学的时变求解。
4. **内存翻倍的冗余 (Memory Bloat)**：为了时间积分，显式将未加边界条件的矩阵拷贝保存（`stiffnessMatrixBeforeBC_`），浪费了大量的内存。

---

### 🛠️ 步骤化重构方案

我们的核心重构思路是**控制反转 (Inversion of Control)**：物理场求解器（Solver）只负责提供 $M, K, F$ 矩阵和边界条件，由通用的外部算法（稳态/时间积分器）自由组合它们。

#### Step 1: 重新定义基类并统一接口 (PhysicsFieldSolver)
将原先大杂烩的 `assemble()` 拆分为按需组装的纯虚函数。这消除了物理场类对自己运行环境（瞬态/稳态）的猜测。

**修改 `physics\physics_field_solver.hpp`**：
```cpp
class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;
    virtual std::string fieldName() const = 0;

    // 核心重构：将组装拆分为标准的数学分量
    virtual void buildStiffnessMatrix(SparseMatrix& K) = 0;
    virtual void buildMassMatrix(SparseMatrix& M) { M.resize(0, 0); } // 默认无质量矩阵
    virtual void buildRHS(Vector& F) = 0;
    virtual void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution) = 0;

    // 通用的线性系统求解
    virtual bool solveLinearSystem(SparseMatrix& A, const Vector& b, Vector& x) {
        A.makeCompressed();
        solver_->setup(&A);
        solver_->apply(b, x);
        field().markUpdated();
        return true;
    }
    // ... 保持原有 getter / setter 不变 ...
};
```

#### Step 2: 瘦身热传导求解器 (HeatTransferSolver)
移除所有与时间步相关的历史包裹（删除 `massMatrixAssembled_`、`stiffnessMatrixBeforeBC_`、`rhsBeforeBC_` 成员），按需提供矩阵。

**修改 `physics\heat_transfer_solver.cpp`**：
```cpp
// 在 initialize 中，删除之前强制调用的 if(!massBindings_.empty()) assembleMassMatrix();

void HeatTransferSolver::buildStiffnessMatrix(SparseMatrix& K) {
    matAsm_->clear(); matAsm_->clearIntegrators();
    for (const auto& binding : conductivityBindings_) {
        matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(binding.conductivity), binding.domains);
    }
    for (const auto& binding : convectionBindings_) {
        for (int bid : binding.boundaryIds) {
            matAsm_->addBoundaryIntegrator(std::make_unique<ConvectionMassIntegrator>(binding.h), bid);
        }
    }
    matAsm_->assemble();
    K = matAsm_->matrix();
}

void HeatTransferSolver::buildMassMatrix(SparseMatrix& M) {
    if (massBindings_.empty()) { M.resize(0, 0); return; }
    BilinearFormAssembler massAsm(fes_.get());
    for (const auto& binding : massBindings_) {
        massAsm.addDomainIntegrator(std::make_unique<MassIntegrator>(binding.thermalMass), binding.domains);
    }
    massAsm.assemble();
    M = massAsm.matrix();
}

void HeatTransferSolver::buildRHS(Vector& F) {
    vecAsm_->clear(); vecAsm_->clearIntegrators();
    for (const auto& binding : heatSourceBindings_) {
        vecAsm_->addDomainIntegrator(std::make_unique<DomainLFIntegrator>(binding.source), binding.domains);
    }
    for (const auto& binding : convectionBindings_) {
        for (int bid : binding.boundaryIds) {
            vecAsm_->addBoundaryIntegrator(std::make_unique<ConvectionLFIntegrator>(binding.h, binding.Tinf), bid);
        }
    }
    vecAsm_->assemble();
    F = vecAsm_->vector();
}

void HeatTransferSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution) {
    std::map<int, const VariableNode*> temperatureBCs;
    for (const auto& binding : temperatureBindings_) {
        for (int bid : binding.boundaryIds) temperatureBCs[bid] = binding.temperature;
    }
    applyDirichletBC(A, rhs, solution, *fes_, *mesh_, temperatureBCs);
}
```
*(注：对于 `ElectrostaticsSolver` 和 `StructuralSolver`，可进行相同的拆分适配重构，使整个物理场模块的底层协议严格一致。)*

#### Step 3: 泛化且解耦时间积分器 (TimeIntegrator)
消除对 `TransientProblem` 的硬依赖，让 TimeIntegrator 回归纯数学组装的本质。以 BDF1 为例。

**修改 `time\time_integrator.hpp` 和 `bdf1_integrator.cpp`**：
```cpp
// time_integrator.hpp
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;
    // 依赖反转：只认识 FieldValues 和通用的 PhysicsFieldSolver，不再认识 Problem
    virtual bool step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) = 0;
// ...
};

// bdf1_integrator.cpp
bool BDF1Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) {
    SparseMatrix M, K;
    Vector F;
    
    solver.buildMassMatrix(M);
    solver.buildStiffnessMatrix(K);
    solver.buildRHS(F);
    
    if (M.rows() == 0 || M.cols() == 0) {
        LOG_ERROR << "BDF1Integrator: Mass matrix is not available for " << solver.fieldName();
        return false;
    }

    ensureSize(M.rows(), M.cols());
    
    const GridFunction& prev = history.history(solver.fieldName(), 1);
    GridFunction& curr = history.current(solver.fieldName());
    
    // 组装广义方程：(M + dt * K) * x_{n+1} = M * x_n + dt * F
    A_ = M + (dt * K);
    rhs_ = M * prev.values() + dt * F;
    
    // 统一施加边界条件并进行系统求解
    solver.applyEssentialBCs(A_, rhs_, curr.values());
    
    if (!solver.solveLinearSystem(A_, rhs_, curr.values())) {
        LOG_ERROR << "BDF1Integrator: Linear solve failed for " << solver.fieldName();
        return false;
    }
    
    return true;
}
```

#### Step 4: 规整系统调用层 (Steady & Transient Problem)
移除业务层的丑陋逻辑，利用接口多态性调用流程。

**修改 `problem\transient_problem.cpp` 中的耦合步骤**：
```cpp
bool solveCouplingStep(TransientProblem& problem, TimeIntegrator& integrator, bool hasElec, bool hasHeat, Real& residual) {
    residual = 0.0;
    Vector prevT;
    
    for (int picardIter = 0; picardIter < problem.couplingMaxIter; ++picardIter) {
        if (hasElec) {
            problem.electrostatics->solveSteady(); // 电磁场是准稳态
        }
        if (hasHeat) {
            // 通过接口直接积分，完全解耦
            if (!integrator.step(*problem.heatTransfer, problem.fieldValues, problem.timeStep, problem.currentStep)) {
                return false;
            }
        }
        
        residual = temperatureResidual(*problem.heatTransfer, prevT, picardIter);
        if (residual < problem.couplingTol) {
            return true;
        }
    }
    return false;
}
```

### 🏆 重构收益总结：
1. **接口一致**：所有 Physics 类的接口完全一致，不再存在专属特供版的 `solveLinearSystem` 和状态缓存，代码极为紧凑简洁。
2. **0无用开销**：稳态程序不会再因为材质设置了密度参数就莫名其妙占用内存去装配 `MassMatrix`，实现真正的懒计算 (Lazy Assembly)。
3. **开闭原则(OCP)**：打破了 `time` 层和 `problem` 层的互相 `#include`。现在如果你要让固体力学（Structural）实现时变动力学求解，时间积分器（BDF1/2）**一行代码都不用改**，直接传入即可复用。
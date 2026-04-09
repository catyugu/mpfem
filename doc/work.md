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

这是一个非常典型的有限元/科学计算代码中的设计反模式：**状态泄露与缺乏缓存（State Leakage & Lack of Caching）**。

在当前的代码中，系统矩阵 $K, M$ 和右端项 $F$ 的生命周期是由调用方（`Problem` 或 `TimeIntegrator`）管理的。这意味着每次迭代都会重新创建矩阵、重新遍历单元进行装配、并强制底层求解器（如 PARDISO/UMFPACK）重新进行代价高昂的符号分解和数值分解（`setup()`），即使对于线性问题，刚度矩阵 $K$ 从头到尾都没变过！

为了彻底解决这个问题，我们需要使用 **模板方法模式（Template Method Pattern）**，将矩阵的状态、装配逻辑和求解器缓存统一封装到 `PhysicsFieldSolver` 基类中，反转依赖关系。

以下是分四步进行的重构计划，每一步都保持编译通过且功能正确。

---

### 第一步：重构 `PhysicsFieldSolver` 基类接口

我们将系统矩阵、RHS 向量以及“脏标记（Dirty Flags）”移入基类。将公开的 `build*` 方法改为受保护的方法，并对外提供高级的 `solveSteady()` 和 `solveTransient()` 接口。

修改 **`physics\physics_field_solver.hpp`**：

```cpp
#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "assembly/assembler.hpp"
#include "core/logger.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "physics/field_values.hpp"
#include "solver/linear_operator.hpp"
#include "solver/solver_config.hpp"
#include <memory>

namespace mpfem {

    class PhysicsFieldSolver {
    public:
        virtual ~PhysicsFieldSolver() = default;

        virtual std::string fieldName() const = 0;

        // 对外提供的统一稳态/准静态求解接口
        virtual bool solveSteady() {
            if (systemMatrixNeedsRebuild_) {
                buildStiffnessMatrix(K_uneliminated_);
                systemMatrixNeedsRebuild_ = false;
                solverNeedsSetup_ = true;
            }

            // 右端项可能包含非线性耦合源项，目前每次都重新装配
            buildRHS(F_);

            // 拷贝未消除的矩阵，并应用边界条件
            K_eliminated_ = K_uneliminated_;
            applyEssentialBCs(K_eliminated_, F_, field().values());

            if (solverNeedsSetup_) {
                K_eliminated_.makeCompressed();
                solver_->setup(&K_eliminated_);
                solverNeedsSetup_ = false;
            }

            solver_->apply(F_, field().values());
            field().markUpdated();
            return true;
        }

        // 对外提供的瞬态求解接口
        virtual bool solveTransient(Real dt, const Vector& historyCombo) {
            bool matrixChanged = systemMatrixNeedsRebuild_;
            if (matrixChanged) {
                buildStiffnessMatrix(K_uneliminated_);
                buildMassMatrix(M_);
                systemMatrixNeedsRebuild_ = false;
            }

            // 如果矩阵变了，或者时间步长变了，重新计算 A = M + dt * K
            if (matrixChanged || std::abs(dt - previous_dt_) > 1e-12) {
                if (M_.rows() == 0) {
                    A_uneliminated_ = dt * K_uneliminated_;
                } else {
                    A_uneliminated_ = M_ + (dt * K_uneliminated_);
                }
                previous_dt_ = dt;
                solverNeedsSetup_ = true;
            }

            buildRHS(F_);
            
            // 组装瞬态右端项: RHS = M * historyCombo + dt * F
            Vector transient_rhs;
            if (M_.rows() > 0) {
                transient_rhs = M_ * historyCombo + (dt * F_);
            } else {
                transient_rhs = dt * F_;
            }

            A_eliminated_ = A_uneliminated_;
            applyEssentialBCs(A_eliminated_, transient_rhs, field().values());

            if (solverNeedsSetup_) {
                A_eliminated_.makeCompressed();
                solver_->setup(&A_eliminated_);
                solverNeedsSetup_ = false;
            }

            solver_->apply(transient_rhs, field().values());
            field().markUpdated();
            return true;
        }

        // 供外部在材料参数改变时调用
        void markMatrixChanged() {
            systemMatrixNeedsRebuild_ = true;
        }

        const GridFunction& field() const { return fieldValues_->current(fieldName()); }
        GridFunction& field() { return fieldValues_->current(fieldName()); }
        const FESpace& feSpace() const { return *fes_; }
        Index numDofs() const { return fes_ ? fes_->numDofs() : 0; }
        const Mesh& mesh() const { return *mesh_; }
        void setSolverConfig(std::unique_ptr<LinearOperatorConfig> config) { solverConfig_ = std::move(config); }
        int iterations() const { return solver_->iterations(); }
        Real residual() const { return solver_->residual(); }

    protected:
        // 内部接口：子类只需负责具体的组装逻辑
        virtual void buildStiffnessMatrix(SparseMatrix& K) = 0;
        virtual void buildMassMatrix(SparseMatrix& M) { M.resize(0, 0); }
        virtual void buildRHS(Vector& F) = 0;
        virtual void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution) = 0;

        int order_ = 1;
        std::unique_ptr<LinearOperatorConfig> solverConfig_;
        const Mesh* mesh_ = nullptr;
        FieldValues* fieldValues_ = nullptr;
        std::unique_ptr<FESpace> fes_;
        std::unique_ptr<BilinearFormAssembler> matAsm_;
        std::unique_ptr<LinearFormAssembler> vecAsm_;
        std::unique_ptr<LinearOperator> solver_;

        // 缓存状态
        SparseMatrix K_uneliminated_, K_eliminated_;
        SparseMatrix M_;
        SparseMatrix A_uneliminated_, A_eliminated_;
        Vector F_;
        
        Real previous_dt_ = -1.0;
        bool systemMatrixNeedsRebuild_ = true;
        bool solverNeedsSetup_ = true;
    };

} // namespace mpfem

#endif
```

---

### 第二步：适配具体的物理场求解器

由于基础控制流（包括调用 `setup()` 和 `apply()`）已经被提到了基类，我们需要从具体的子类中删除 `solveLinearSystem` 等冗余的公有方法，并将 `build*` 方法改为受保护的 `protected` 访问权限。

请对 **`electrostatics_solver.hpp`**, **`heat_transfer_solver.hpp`**, **`structural_solver.hpp`** 进行相同的修改：
1. 删除 `solveLinearSystem` 的声明。
2. 将 `buildStiffnessMatrix`、`buildMassMatrix`、`buildRHS`、`applyEssentialBCs` 的访问权限移至 `protected` 区域。

同时，在对应的 **`.cpp` 文件** 中：
删除 `ElectrostaticsSolver::solveLinearSystem`、`HeatTransferSolver::solveLinearSystem`、`StructuralSolver::solveLinearSystem` 的实现代码（因为逻辑已经被基类的 `solveSteady` 和 `solveTransient` 取代）。

---

### 第三步：重构时间积分器 (Time Integrator)

现在，时间积分器不需要自己维护矩阵或计算 `A = M + dt * K` 了。它只需要计算历史项组合（`historyCombo`），并传给 `PhysicsFieldSolver` 即可。

修改 **`time\time_integrator.hpp`**：
删除 `A_`, `rhs_`, `initialized_`, `ensureSize()`。使其成为一个纯粹的接口。
```cpp
    class TimeIntegrator {
    public:
        virtual ~TimeIntegrator() = default;
        virtual bool step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) = 0;
    };
```

修改 **`time\bdf1_integrator.cpp`**：
```cpp
#include "time/bdf1_integrator.hpp"
#include "core/logger.hpp"

namespace mpfem {

    bool BDF1Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        const GridFunction& prev = history.history(solver.fieldName(), 1);
        
        // 对于 BDF1，历史项组合就是前一步的值
        const Vector historyCombo = prev.values();

        if (!solver.solveTransient(dt, historyCombo)) {
            LOG_ERROR << "BDF1Integrator: Transient solve failed for " << solver.fieldName();
            return false;
        }

        LOG_INFO << "BDF1Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
```

修改 **`time\bdf2_integrator.cpp`**：
```cpp
#include "time/bdf2_integrator.hpp"
#include "core/logger.hpp"

namespace mpfem {

    bool BDF2Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        Vector historyCombo;
        
        if (currentStep > 0) {
            // BDF2 formula: (2 * u_{n} - 0.5 * u_{n-1})
            const GridFunction& prev1 = history.history(solver.fieldName(), 1);
            const GridFunction& prev2 = history.history(solver.fieldName(), 2);
            historyCombo = 2.0 * prev1.values() - 0.5 * prev2.values();
            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF2)";
        } else {
            // 第一个时间步退化为 BDF1
            const GridFunction& prev = history.history(solver.fieldName(), 1);
            historyCombo = prev.values();
            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF1 starter)";
        }

        if (!solver.solveTransient(dt, historyCombo)) {
            LOG_ERROR << "BDF2Integrator: Transient solve failed for " << solver.fieldName();
            return false;
        }

        LOG_INFO << "BDF2Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
```

---

### 第四步：清理 Problem 类调用端

现在最激动人心的地方来了。我们可以在 `SteadyProblem` 和 `TransientProblem` 中把冗长的矩阵分配全部删掉，调用代码变得极度精简！

修改 **`problem\steady_problem.hpp`** 的 `solve()` 方法：

```cpp
        SteadyResult solve()
        {
            ScopedTimer timer("Coupling solve");
            SteadyResult result;

            if (!isCoupled()) {
                if (hasElectrostatics()) electrostatics->solveSteady();
                if (hasHeatTransfer()) heatTransfer->solveSteady();
                if (hasStructural()) structural->solveSteady();
                result.fields = fieldValues;
                return result;
            }

            const bool hasE = hasElectrostatics();
            const bool hasT = hasHeatTransfer();

            for (int i = 0; i < couplingMaxIter; ++i) {
                if (hasE) electrostatics->solveSteady();
                if (hasT) heatTransfer->solveSteady();

                Real err = computeCouplingError();
                result.iterations = i + 1;
                result.residual = err;
                LOG_INFO << "Coupling iteration " << (i + 1) << ", residual = " << err;
                if (err < couplingTol) {
                    result.converged = true;
                    break;
                }
                
                // 【注意】如果材料参数(如电导率)依赖于温度，此处应该标记矩阵需要重建：
                // if (hasE) electrostatics->markMatrixChanged(); 
            }

            if (hasStructural()) structural->solveSteady();
            
            result.fields = fieldValues;
            return result;
        }
```

修改 **`problem\transient_problem.cpp`** 中辅助函数 `solveCouplingStep`：

```cpp
        bool solveCouplingStep(TransientProblem& problem,
            TimeIntegrator& integrator,
            bool hasElectrostatics,
            bool hasHeatTransfer,
            Real& residual)
        {
            residual = 0.0;
            if (!hasHeatTransfer) {
                // 如果没有热传递，电场视为准静态求解
                if (hasElectrostatics) problem.electrostatics->solveSteady();
                return true;
            }

            Vector prevT;
            for (int picardIter = 0; picardIter < problem.couplingMaxIter; ++picardIter) {
                // 准静态电场
                if (hasElectrostatics) problem.electrostatics->solveSteady();

                // 瞬态热传导
                if (!integrator.step(*problem.heatTransfer, problem.fieldValues, problem.timeStep, problem.currentStep)) {
                    LOG_ERROR << "TransientProblem::solve: Time step failed";
                    return false;
                }

                residual = temperatureResidual(*problem.heatTransfer, prevT, picardIter);
                LOG_INFO << "  Picard iter " << (picardIter + 1) << ", T residual = " << residual;

                if (residual < problem.couplingTol) {
                    return true;
                }
                // 如有强非线性耦合，视情况取消注释: problem.electrostatics->markMatrixChanged();
            }
            return false;
        }
```

修改 **`problem\transient_problem.cpp`** 中 `initializeSteadyState`：
```cpp
    void TransientProblem::initializeSteadyState()
    {
        LOG_INFO << "Steady-state initialization at t=0";
        if (hasElectrostatics()) electrostatics->solveSteady();
        if (hasStructural()) structural->solveSteady();
        LOG_INFO << "Steady-state initialization complete";
    }
```
再修改 `solve` 尾部的力学计算：
```cpp
            if (hasStructural) {
                structural->solveSteady();
            }
```

### 重构收益总结：
1. **彻底消除泄漏：** `SparseMatrix` 和 `Vector` 不再在最高层的业务代码（Problem）中满天飞，降低了内存申请和移动的开销。
2. **获得极高收益的缓存（极速求解）：** `K_uneliminated_` 只会在第一步计算一次。接下来如果调用 `solveSteady()` 或时间积分器中 `dt` 不变，底层的 `PardisoSolver` 或 `UmfpackSolver` 会**直接跳过符号分解和数值分解（Factorization）**，纯粹只做前代后代回代（Forward/Backward substitution），性能可提升数十倍。
3. **接口极度统一：** 时间积分算法只需要关心 `historyCombo`（$2u_n - 0.5u_{n-1}$ 等），而不需要关心偏微分方程是怎么变成矩阵的。这完美符合了科学计算软件分层解耦的黄金法则。
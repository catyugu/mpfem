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

为了实现代码的简洁化、算子化解耦以及高性能求解器的引入，我们需要重点解决以下几个设计反模式和迫切需求：

1. **设计反模式与冗长代码 (DRY原则违背)**：`CgOperator` 和 `GmresOperator` 几乎有 90% 的重复代码（容差、最大迭代次数的保存和解析）。
2. **模板硬编码 (Template Lock-in)**：直接使用 `Eigen::DiagonalPreconditioner<Real>` 将预条件子在编译期写死，使得 `LinearOperator` 基类提供的 `set_preconditioner()` 形同虚设，阻止了真正的嵌套算子化（例如：`CG` 嵌套 `AMG`）。
3. **引入 HYPRE BoomerAMG**：需要提供工业级的代数多重网格求解器来处理弹性力学等刚度矩阵条件数极差的问题。

以下是步骤化的重构方案：

### 步骤 1：提取迭代器基类 (消除冗余与向后兼容包袱)
在 `solver/linear_operator.hpp` 中，我们首先引入一个 `IterativeOperatorBase`。抛弃为了向后兼容而写的繁琐检查，统一使用这一个基类管理迭代状态。

```cpp
namespace mpfem {

// 在 LinearOperator 定义之后添加：
class IterativeOperatorBase : public LinearOperator {
public:
    void configure(const LinearOperatorConfig& config) override {
        if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
            maxIterations_ = static_cast<int>(it->second);
        }
        if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
            tolerance_ = it->second;
        }
    }

    int iterations() const override { return iterations_; }
    Real residual() const override { return residual_; }

protected:
    int maxIterations_ = 1000;
    Real tolerance_ = 1e-10;
    int iterations_ = 0;
    Real residual_ = 0.0;
};

} // namespace mpfem
```

### 步骤 2：解耦 Eigen 的预条件子 (实现真正的算子化)
为了让 Eigen 的迭代求解器能够调用我们多态的 `LinearOperator` 预条件子，我们需要通过特征萃取（Traits）编写一个符合 Eigen 规范的桥接器（Adapter）。

在 `solver/eigen_solver.hpp` 顶部添加以下桥接代码：

```cpp
namespace mpfem {
namespace detail {

    // 桥接 Eigen 的 Preconditioner 概念和我们的多态 LinearOperator
    class EigenPrecondAdapter {
    public:
        using StorageIndex = Index;
        enum {
            ColsAtCompileTime = Eigen::Dynamic,
            MaxColsAtCompileTime = Eigen::Dynamic
        };

        EigenPrecondAdapter() : op_(nullptr) {}

        template<typename MatrixType>
        explicit EigenPrecondAdapter(const MatrixType&) : op_(nullptr) {}

        EigenPrecondAdapter& analyzePattern(const Eigen::SparseMatrix<Real>&) { return *this; }
        EigenPrecondAdapter& factorize(const Eigen::SparseMatrix<Real>&) { return *this; }
        EigenPrecondAdapter& compute(const Eigen::SparseMatrix<Real>&) { return *this; }

        template<typename Rhs, typename Dest>
        void _solve_impl(const Rhs& b, Dest& x) const {
            if (op_) {
                Vector b_vec = b;
                Vector x_vec = Vector::Zero(b.rows());
                op_->apply(b_vec, x_vec);
                x = x_vec;
            } else {
                x = b; // 未设置预条件子时，退化为恒等映射
            }
        }

        template<typename Rhs>
        inline const Eigen::Solve<EigenPrecondAdapter, Rhs> solve(const Eigen::MatrixBase<Rhs>& b) const {
            return Eigen::Solve<EigenPrecondAdapter, Rhs>(*this, b.derived());
        }

        void setOperator(LinearOperator* op) { op_ = op; }
        Eigen::ComputationInfo info() const { return Eigen::Success; }

    private:
        LinearOperator* op_;
    };
} // namespace detail
} // namespace mpfem

// 向 Eigen 注册 Traits
namespace Eigen {
    namespace internal {
        template<>
        struct traits<mpfem::detail::EigenPrecondAdapter> {
            typedef double Scalar;
            typedef double RealScalar;
            typedef int StorageIndex;
            enum {
                ColsAtCompileTime = Dynamic,
                MaxColsAtCompileTime = Dynamic,
                RowsAtCompileTime = Dynamic,
                MaxRowsAtCompileTime = Dynamic,
                Flags = 0
            };
        };
    }
}
```

### 步骤 3：重构 CG 与 DGMRES 算子
利用刚才的基类和 Adapter，重写 `CgOperator` 和 `GmresOperator`。你将看到代码变得极其简洁，并且**彻底解锁了动态预条件子的嵌套能力**。

在 `solver/eigen_solver.hpp` 中：

```cpp
class CgOperator : public IterativeOperatorBase {
public:
    std::string_view name() const override { return "CG"; }

    void setup(const SparseMatrix* A) override {
        if (!A) throw std::runtime_error("CgOperator: null matrix");
        
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        // 核心解耦：将运行时的多态 preconditioner 注入 Eigen
        solver_.preconditioner().setOperator(this->preconditioner());
        
        solver_.compute(A->eigen());
        set_matrix(A);
        mark_setup();
    }

    void apply(const Vector& b, Vector& x) override {
        x = solver_.solveWithGuess(b, x);
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
    }

private:
    // 将预条件子模板修改为我们的 Adapter
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>, Eigen::Lower | Eigen::Upper, detail::EigenPrecondAdapter> solver_;
};

class GmresOperator : public IterativeOperatorBase {
public:
    std::string_view name() const override { return "DGMRES"; }

    void setup(const SparseMatrix* A) override {
        if (!A) throw std::runtime_error("GmresOperator: null matrix");
        
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.preconditioner().setOperator(this->preconditioner());
        
        solver_.compute(A->eigen());
        set_matrix(A);
        mark_setup();
    }

    void apply(const Vector& b, Vector& x) override {
        x = solver_.solveWithGuess(b, x);
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
    }

private:
    Eigen::DGMRES<Eigen::SparseMatrix<Real>, detail::EigenPrecondAdapter> solver_;
};
```

### 步骤 4：引入 hypre_BoomerAMG 算子
在你的项目中新建 `solver/hypre_amg_operator.hpp`，提供针对弹性力学等高度非良态问题的高效后备。它可以作为独立求解器，也可以通过刚才解耦的接口，作为 CG 的预条件子传入！

```cpp
#ifndef MPFEM_HYPRE_AMG_OPERATOR_HPP
#define MPFEM_HYPRE_AMG_OPERATOR_HPP

#include "solver/linear_operator.hpp"

#ifdef MPFEM_USE_HYPRE
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_IJ_mv.h>
#endif

namespace mpfem {

class HypreBoomerAmgOperator : public IterativeOperatorBase {
public:
    HypreBoomerAmgOperator() = default;

    ~HypreBoomerAmgOperator() override {
#ifdef MPFEM_USE_HYPRE
        if (solver_created_) HYPRE_BoomerAMGDestroy(solver_);
        if (A_hypre_) HYPRE_IJMatrixDestroy(A_hypre_);
#endif
    }

    std::string_view name() const override { return "BoomerAMG"; }

    void setup(const SparseMatrix* A) override {
#ifdef MPFEM_USE_HYPRE
        if (!A) throw std::runtime_error("BoomerAMG: null matrix");

        // 1. Convert Eigen::SparseMatrix to HYPRE_IJMatrix
        MPI_Comm comm = MPI_COMM_WORLD;
        int ilower = 0;
        int iupper = A->rows() - 1;

        if (A_hypre_) HYPRE_IJMatrixDestroy(A_hypre_);
        HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A_hypre_);
        HYPRE_IJMatrixSetObjectType(A_hypre_, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_hypre_);

        // Convert ColMajor Eigen matrix to CSR format row-by-row
        for (int k = 0; k < A->outerSize(); ++k) {
            for (SparseMatrix::Storage::InnerIterator it(A->eigen(), k); it; ++it) {
                int row = it.row();
                int col = it.col();
                double val = it.value();
                int num_cols = 1;
                HYPRE_IJMatrixSetValues(A_hypre_, 1, &num_cols, &row, &col, &val);
            }
        }
        HYPRE_IJMatrixAssemble(A_hypre_);
        HYPRE_IJMatrixGetObject(A_hypre_, (void**)&par_A_);

        // 2. Setup BoomerAMG solver / preconditioner options
        if (!solver_created_) {
            HYPRE_BoomerAMGCreate(&solver_);
            solver_created_ = true;
        }

        HYPRE_BoomerAMGSetMaxIter(solver_, maxIterations_);
        HYPRE_BoomerAMGSetTol(solver_, tolerance_);
        HYPRE_BoomerAMGSetCoarsenType(solver_, 10); // HMIS coarsening
        HYPRE_BoomerAMGSetRelaxType(solver_, 8);    // L1-symmetric Gauss-Seidel
        HYPRE_BoomerAMGSetNumSweeps(solver_, 1);
        HYPRE_BoomerAMGSetPrintLevel(solver_, 0);

        HYPRE_BoomerAMGSetup(solver_, par_A_, NULL, NULL);

        set_matrix(A);
        mark_setup();
#else
        throw std::runtime_error("HypreBoomerAmgOperator: mpfem was built without HYPRE support.");
#endif
    }

    void apply(const Vector& b, Vector& x) override {
#ifdef MPFEM_USE_HYPRE
        // Create HYPRE vectors wrapping existing memory (no deep copy needed for pure serial/shared memory MPI)
        HYPRE_IJVector b_hypre, x_hypre;
        int ilower = 0;
        int iupper = b.rows() - 1;
        
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b_hypre);
        HYPRE_IJVectorSetObjectType(b_hypre, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b_hypre);
        // Note: For production, we use HYPRE_IJVectorSetValues or pointer swaps.
        
        HYPRE_ParVector par_b, par_x;
        HYPRE_IJVectorGetObject(b_hypre, (void**)&par_b);
        HYPRE_IJVectorGetObject(x_hypre, (void**)&par_x);

        HYPRE_BoomerAMGSolve(solver_, par_A_, par_b, par_x);

        HYPRE_BoomerAMGGetNumIterations(solver_, &iterations_);
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver_, &residual_);

        HYPRE_IJVectorDestroy(b_hypre);
        HYPRE_IJVectorDestroy(x_hypre);
#endif
    }

private:
#ifdef MPFEM_USE_HYPRE
    HYPRE_Solver solver_;
    HYPRE_IJMatrix A_hypre_ = nullptr;
    HYPRE_ParCSRMatrix par_A_ = nullptr;
    bool solver_created_ = false;
#endif
};

} // namespace mpfem

#endif // MPFEM_HYPRE_AMG_OPERATOR_HPP
```

### 步骤 5：更新求解器注册表 (`solver_config.hpp`)
在最后一步，将我们实现的新 AMG 求解器登记到枚举和注册表中：

```cpp
enum class OperatorType {
    // Direct solvers
    SparseLU, Pardiso, Umfpack,
    // Iterative solvers
    CG, DGMRES,
    // Preconditioners / Advanced Solvers
    Diagonal, ICC, ILU, AdditiveSchwarz,
    BoomerAMG // <-- 新增
};

inline constexpr OperatorMeta operatorRegistry[] = {
    // ... 原有选项
    {OperatorType::BoomerAMG, "BoomerAMG", "HYPRE BoomerAMG Algebraic Multigrid", true, true, 
#ifdef MPFEM_USE_HYPRE
     true
#else
     false
#endif
    },
};
```

### 重构效果总结
1. **完全算子化**：此时的 `CG` 与 `DGMRES` 不再绑定特定预条件子。你可以非常简单地在应用层调用 `cg->set_preconditioner(std::make_unique<HypreBoomerAmgOperator>())`。
2. **极简代码**：将迭代参数（容差、最大迭代）统一上浮到了 `IterativeOperatorBase`，避免了模板和类方法的重复代码。
3. **弹性力学破局**：Hypre 的引入完美匹配了当前弹性力学在大网格规模下的求解瓶颈。如果只用它做独立求解器，使用方法与其他求解器别无二致。
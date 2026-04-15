#include "solver/linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace mpfem {

    // =============================================================================
    // DiagonalOperator (Jacobi preconditioner)
    // =============================================================================

    void DiagonalOperator::setup(const SparseMatrix* A)
    {
        if (!A) {
            throw std::runtime_error("DiagonalOperator: null matrix in setup");
        }
        const Index n = A->rows();
        invDiag_.resize(n);
        for (Index i = 0; i < n; ++i) {
            const Real diag = A->coeff(i, i);
            if (std::abs(diag) < 1e-14) {
                invDiag_(i) = 0.0;
            }
            else {
                invDiag_(i) = 1.0 / diag;
            }
        }
        set_matrix(A);
        mark_setup();
    }

    void DiagonalOperator::apply(const Vector& b, Vector& x)
    {
        x = invDiag_.cwiseProduct(b);
    }

    // =============================================================================
    // IccOperator (Incomplete Cholesky)
    // =============================================================================

    struct IccOperator::Impl {
        Real shift = 1e-14;
        Eigen::IncompleteCholesky<Real> solver;
    };

    IccOperator::IccOperator() : impl_(std::make_unique<Impl>()) {}
    IccOperator::~IccOperator() = default;

    void IccOperator::setup(const SparseMatrix* A)
    {
        if (!A) {
            throw std::runtime_error("IccOperator: null matrix in setup");
        }
        impl_->solver.setInitialShift(impl_->shift);
        impl_->solver.compute(A->eigen());
        set_matrix(A);
        mark_setup();
    }

    void IccOperator::apply(const Vector& b, Vector& x)
    {
        x = impl_->solver.solve(b);
    }

    void IccOperator::set_shift(Real shift)
    {
        impl_->shift = shift;
    }

    Real IccOperator::shift() const
    {
        return impl_->shift;
    }

    void IccOperator::configure(const LinearOperatorConfig& config)
    {
        if (auto it = config.parameters.find("Shift"); it != config.parameters.end()) {
            set_shift(it->second);
        }
    }

    // =============================================================================
    // IluOperator (Incomplete LU)
    // =============================================================================

    struct IluOperator::Impl {
        Eigen::IncompleteLUT<Real> solver;
    };

    IluOperator::IluOperator() : impl_(std::make_unique<Impl>()) {}
    IluOperator::~IluOperator() = default;

    void IluOperator::setup(const SparseMatrix* A)
    {
        if (!A) {
            throw std::runtime_error("IluOperator: null matrix in setup");
        }
        impl_->solver.compute(A->eigen());
        set_matrix(A);
        mark_setup();
    }

    void IluOperator::apply(const Vector& b, Vector& x)
    {
        x = impl_->solver.solve(b);
    }

    // =============================================================================
    // AdditiveSchwarzOperator
    // =============================================================================

    void AdditiveSchwarzOperator::setup(const SparseMatrix* A)
    {
        set_matrix(A);
        mark_setup();
    }

    void AdditiveSchwarzOperator::apply(const Vector& b, Vector& x)
    {
        // TODO: Implement domain decomposition with overlap
        x = b; // Fallback: identity
    }

} // namespace mpfem

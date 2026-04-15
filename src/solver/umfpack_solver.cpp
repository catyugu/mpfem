#include "solver/umfpack_solver.hpp"
#include <stdexcept>

#ifdef MPFEM_USE_UMFPACK
#include <Eigen/UmfPackSupport>
#endif

namespace mpfem {

#ifdef MPFEM_USE_UMFPACK

    struct UmfpackSolver::Impl {
        Eigen::UmfPackLU<SparseMatrix::Storage> solver;
        int iterations = 1;
        Real residual = 0.0;
    };

    UmfpackSolver::UmfpackSolver() : impl_(std::make_unique<Impl>()) { }
    UmfpackSolver::~UmfpackSolver() = default;

    void UmfpackSolver::setup(const SparseMatrix* A)
    {
        if (!A) throw std::runtime_error("UmfpackSolver: null matrix in setup");
        impl_->solver.compute(A->eigen());
        if (impl_->solver.info() != Eigen::Success) {
            throw std::runtime_error("UmfpackSolver: factorization failed");
        }
        set_matrix(A);
        mark_setup();
    }

    void UmfpackSolver::apply(const Vector& b, Vector& x)
    {
        if (!is_setup()) throw std::runtime_error("UmfpackSolver: not setup");
        x = impl_->solver.solve(b);
        if (impl_->solver.info() != Eigen::Success) {
            throw std::runtime_error("UmfpackSolver: solve failed");
        }
        impl_->iterations = 1;
        impl_->residual = 0.0;
    }

    int UmfpackSolver::iterations() const { return impl_->iterations; }
    Real UmfpackSolver::residual() const { return impl_->residual; }

#else

    struct UmfpackSolver::Impl { };
    UmfpackSolver::UmfpackSolver() { throw std::runtime_error("UmfpackSolver: SuiteSparse not available"); }
    UmfpackSolver::~UmfpackSolver() = default;
    void UmfpackSolver::setup(const SparseMatrix*) { throw std::runtime_error("UmfpackSolver: SuiteSparse not available"); }
    void UmfpackSolver::apply(const Vector&, Vector&) { throw std::runtime_error("UmfpackSolver: SuiteSparse not available"); }
    int UmfpackSolver::iterations() const { return 0; }
    Real UmfpackSolver::residual() const { return 0.0; }

#endif

} // namespace mpfem

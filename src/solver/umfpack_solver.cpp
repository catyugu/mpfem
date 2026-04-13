#include "solver/umfpack_solver.hpp"

namespace mpfem {

#ifdef MPFEM_USE_UMFPACK

    void UmfpackSolver::setup(const SparseMatrix* A)
    {
        if (!A) {
            throw std::runtime_error("UmfpackSolver: null matrix in setup");
        }

        solver_.analyzePattern(A->eigen());
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("UmfpackSolver: symbolic analysis failed");
        }

        solver_.factorize(A->eigen());

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("UmfpackSolver: factorization failed");
        }

        set_matrix(A);
        mark_setup();
    }

    void UmfpackSolver::apply(const Vector& b, Vector& x)
    {
        if (!is_setup()) {
            throw std::runtime_error("UmfpackSolver: not setup");
        }

        x = solver_.solve(b);

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("UmfpackSolver: solve failed");
        }

        iterations_ = 1;
        residual_ = 0.0;
    }

    std::string_view UmfpackSolver::name() const
    {
        return "UMFPACK";
    }

    int UmfpackSolver::iterations() const
    {
        return iterations_;
    }

    Real UmfpackSolver::residual() const
    {
        return residual_;
    }

#else

    // Stub implementations stay inline in header

#endif

} // namespace mpfem
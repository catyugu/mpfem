#ifndef MPFEM_UMFPACK_SOLVER_HPP
#define MPFEM_UMFPACK_SOLVER_HPP

#include "core/logger.hpp"
#include "linear_operator.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef MPFEM_USE_SUITESPARSE
#include <Eigen/UmfPackSupport>
#endif

namespace mpfem {

#ifdef MPFEM_USE_SUITESPARSE

    /**
     * @brief SuiteSparse UMFPACK direct solver.
     *
     * High-performance direct LU solver from SuiteSparse.
     * Good alternative when MKL PARDISO is not available.
     */
    class UmfpackSolver : public LinearOperator {
    public:
        UmfpackSolver() = default;

        std::string_view name() const override { return "UMFPACK"; }

        void setup(const SparseMatrix* A) override
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

        void apply(const Vector& b, Vector& x) override
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

        int iterations() const override { return iterations_; }
        Real residual() const override { return residual_; }

    private:
        Eigen::UmfPackLU<SparseMatrix::Storage> solver_;
        int iterations_ = 1;
        Real residual_ = 0.0;
    };

#else

    // Stub for when SuiteSparse is not available
    class UmfpackSolver : public LinearOperator {
    public:
        UmfpackSolver()
        {
            throw std::runtime_error("UmfpackSolver: SuiteSparse not available. "
                                     "Rebuild with SuiteSparse support enabled.");
        }
        std::string_view name() const override { return "UMFPACK"; }
        void setup(const SparseMatrix*) override
        {
            throw std::runtime_error("UmfpackSolver: SuiteSparse not available");
        }
        void apply(const Vector&, Vector&) override
        {
            throw std::runtime_error("UmfpackSolver: SuiteSparse not available");
        }
    };

#endif // MPFEM_USE_SUITESPARSE

} // namespace mpfem

#endif // MPFEM_UMFPACK_SOLVER_HPP
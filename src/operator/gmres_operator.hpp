#ifndef MPFEM_GMRES_OPERATOR_HPP
#define MPFEM_GMRES_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <stdexcept>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

    /**
     * @brief DGMRES (Direct GMRES) operator.
     *
     * Iterative solver for general (non-symmetric) matrices.
     * Caches factorization via matrix fingerprint to avoid recompute.
     */
    class GMRESOperator : public LinearOperator {
    public:
        GMRESOperator() { operator_name_ = "GMRES"; }

        void set_parameters(const ParameterList& params) override
        {
            if (params.has("MaxIterations")) {
                solver_.setMaxIterations(params.get_int("MaxIterations"));
            }
            if (params.has("Tolerance")) {
                solver_.setTolerance(params.get_double("Tolerance"));
            }
        }

        void setup(const SparseMatrix* A) override
        {
            // Skip if matrix fingerprint unchanged (cache hit)
            if (is_setup_ && A_ != nullptr) {
                std::uint64_t currentFingerprint = A->fingerprint();
                if (currentFingerprint == lastMatrixFingerprint_) {
                    return; // Reuse existing factorization
                }
            }

            A_ = A;
            solver_.compute(A->eigen());
            lastMatrixFingerprint_ = A->fingerprint();
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("GMRESOperator: setup() must be called before apply()");
            }
            x = solver_.solveWithGuess(b, x);
            num_iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        int num_iterations() const override { return num_iterations_; }
        Real residual() const override { return residual_; }

    private:
        Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::DiagonalPreconditioner<Real>> solver_;
        std::uint64_t lastMatrixFingerprint_ = 0;
        int num_iterations_ = 0;
        Real residual_ = 0.0;
    };

} // namespace mpfem

#endif // MPFEM_GMRES_OPERATOR_HPP
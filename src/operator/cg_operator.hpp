#ifndef MPFEM_CG_OPERATOR_HPP
#define MPFEM_CG_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Conjugate Gradient operator.
     *
     * Iterative solver for symmetric positive definite (SPD) matrices.
     * Uses solveWithGuess() to leverage initial guess from previous iteration.
     * Caches factorization via matrix timestamp to avoid recompute.
     */
    class CGOperator : public LinearOperator {
    public:
        CGOperator() { operator_name_ = "CG"; }

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
            // Skip if matrix timestamp unchanged (cache hit) - O(1) check
            if (is_setup_ && A->timestamp() == lastMatrixTimestamp_) {
                return; // Reuse existing factorization
            }

            A_ = A;
            solver_.compute(A->eigen());
            lastMatrixTimestamp_ = A->timestamp();
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("CGOperator: setup() must be called before apply()");
            }
            // Use solveWithGuess to leverage initial guess from previous Picard iteration
            x = solver_.solveWithGuess(b, x);
            num_iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        int num_iterations() const override { return num_iterations_; }
        Real residual() const override { return residual_; }

    private:
        Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Real>>
            solver_;
        std::uint64_t lastMatrixTimestamp_ = 0;
        int num_iterations_ = 0;
        Real residual_ = 0.0;
    };

} // namespace mpfem

#endif // MPFEM_CG_OPERATOR_HPP
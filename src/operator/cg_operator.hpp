#ifndef MPFEM_CG_OPERATOR_HPP
#define MPFEM_CG_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Lightweight adapter that delegates to a LinearOperator.
     *
     * This adapter satisfies Eigen's preconditioner concept while
     * delegating to our runtime-configurable LinearOperator via
     * a captured pointer (not global storage).
     */
    template <typename MatrixType>
    class EigenPreconditionerAdapter {
    public:
        typedef typename MatrixType::Scalar Scalar;
        typedef typename MatrixType::RealScalar RealScalar;

    private:
        LinearOperator* preconditioner_ = nullptr;

    public:
        EigenPreconditionerAdapter() = default;

        void set_preconditioner(LinearOperator* pc)
        {
            preconditioner_ = pc;
        }

        LinearOperator* get_preconditioner() const
        {
            return preconditioner_;
        }

        template <typename MatType>
        EigenPreconditionerAdapter& compute(const MatType& /*mat*/)
        {
            return *this;
        }

        template <typename Rhs>
        Rhs solve(const Rhs& b) const
        {
            if (preconditioner_) {
                Rhs x;
                x.resize(b.size());
                preconditioner_->apply(b, x);
                return x;
            }
            else {
                return b;
            }
        }

        Eigen::ComputationInfo info() const
        {
            return Eigen::Success;
        }
    };

    /**
     * @brief Conjugate Gradient operator.
     *
     * Iterative solver for symmetric positive definite (SPD) matrices.
     * Supports custom preconditioner via inner_operator_.
     * If no preconditioner is set, uses diagonal preconditioning.
     * Uses solveWithGuess() to leverage initial guess from previous iteration.
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

            // Setup preconditioner if present
            if (inner_operator_) {
                inner_operator_->setup(A);
                preconditionerAdapter_.set_preconditioner(inner_operator_);
            }

            solver_.compute(A->eigen());
            lastMatrixTimestamp_ = A->timestamp();
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("CGOperator: setup() must be called before apply()");
            }

            // Update preconditioner pointer in solver's internal copy
            // (Eigen stores preconditioner by value, so we must update after compute)
            if (inner_operator_) {
                solver_.preconditioner().set_preconditioner(inner_operator_);
            }

            // Use solveWithGuess to leverage initial guess from previous Picard iteration
            x = solver_.solveWithGuess(b, x);
            num_iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        int num_iterations() const override { return num_iterations_; }
        Real residual() const override { return residual_; }

    private:
        using PrecondAdapter = EigenPreconditionerAdapter<Eigen::SparseMatrix<Real>>;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
            Eigen::Lower | Eigen::Upper,
            PrecondAdapter>
            solver_;
        PrecondAdapter preconditionerAdapter_;
        std::uint64_t lastMatrixTimestamp_ = 0;
        int num_iterations_ = 0;
        Real residual_ = 0.0;
    };

} // namespace mpfem

#endif // MPFEM_CG_OPERATOR_HPP
#ifndef MPFEM_ICC_OPERATOR_HPP
#define MPFEM_ICC_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Incomplete Cholesky (ICC) operator.
     *
     * Approximate Cholesky factorization for SPD matrices.
     * Uses Eigen::IncompleteCholesky with configurable shift (damping).
     *
     * Can be used as a preconditioner or standalone solver.
     */
    class ICCOperator : public LinearOperator {
    public:
        ICCOperator()
        {
            operator_name_ = "ICC";
        }

        void set_parameters(const ParameterList& params) override
        {
            if (params.has("shift")) {
                shift_ = params.get_double("shift");
            }
        }

        void setup(const SparseMatrix* A) override
        {
            A_ = A;
            precond_.setInitialShift(shift_);
            precond_.compute(A->eigen());
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("ICCOperator: setup() must be called before apply()");
            }
            x = precond_.solve(b);
        }

    private:
        Eigen::IncompleteCholesky<Real> precond_;
        Real shift_ = 1e-14;
    };

} // namespace mpfem

#endif // MPFEM_ICC_OPERATOR_HPP
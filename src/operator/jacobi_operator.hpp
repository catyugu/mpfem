#ifndef MPFEM_JACOBI_OPERATOR_HPP
#define MPFEM_JACOBI_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Jacobi (diagonal) preconditioner/smoother.
     */
    class JacobiOperator : public LinearOperator {
    public:
        JacobiOperator() { operator_name_ = "Jacobi"; }

        void set_parameters(const ParameterList& params) override
        {
            if (params.has("NumSweeps")) {
                num_sweeps_ = params.get_int("NumSweeps");
            }
        }

        void setup(const SparseMatrix* A) override
        {
            A_ = A;
            const auto& diag = A->eigen().diagonal();
            inverse_diagonal_.resize(diag.size());
            for (int i = 0; i < diag.size(); ++i) {
                inverse_diagonal_(i) = (std::abs(diag(i)) > 1e-12) ? 1.0 / diag(i) : 1.0;
            }
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("JacobiOperator: setup() must be called before apply()");
            }

            for (int sweep = 0; sweep < num_sweeps_; ++sweep) {
                Vector temp = b - A_->eigen() * x;
                x += inverse_diagonal_.cwiseProduct(temp);
            }
        }

    private:
        int num_sweeps_ = 1;
        Vector inverse_diagonal_;
    };

} // namespace mpfem

#endif // MPFEM_JACOBI_OPERATOR_HPP
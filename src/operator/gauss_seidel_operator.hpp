#ifndef MPFEM_GAUSS_SEIDEL_OPERATOR_HPP
#define MPFEM_GAUSS_SEIDEL_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Gauss-Seidel operator.
     *
     * Sequential iterative method for solving linear systems.
     * Each component x_i is updated using previously computed values.
     *
     * Forward sweep: x_i = (b_i - sum_{j<i} A_ij * x_j - sum_{j>i} A_ij * old_x_j) / A_ii
     *
     * Can be used as a smoother in multigrid. Unlike Jacobi, Gauss-Seidel
     * uses updated values immediately, leading to faster convergence for
     * many problems.
     */
    class GaussSeidelOperator : public LinearOperator {
    public:
        GaussSeidelOperator()
        {
            operator_name_ = "GaussSeidel";
        }

        void set_parameters(const ParameterList& params) override
        {
            if (params.has("NumSweeps")) {
                num_sweeps_ = params.get_int("NumSweeps");
            }
        }

        void setup(const SparseMatrix* A) override
        {
            A_ = A;
            // Pre-extract diagonal for efficiency - avoid repeated coeff() calls
            const auto& mat = A_->eigen();
            const Index n = mat.rows();
            diagonal_.resize(n);
            for (Index i = 0; i < n; ++i) {
                diagonal_(i) = mat.coeff(i, i);
                if (std::abs(diagonal_(i)) < 1e-12) {
                    diagonal_(i) = 1.0; // Avoid division by zero
                }
            }
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("GaussSeidelOperator: setup() must be called before apply()");
            }

            const auto& mat = A_->eigen();
            const Index n = mat.rows();

            for (int sweep = 0; sweep < num_sweeps_; ++sweep) {
                // Row-wise iteration - efficient for Gauss-Seidel forward sweep
                for (Index i = 0; i < n; ++i) {
                    Real sum = b(i);

                    // Iterate only over non-zeros in row i
                    for (SparseMatrix::Storage::InnerIterator it(mat, i); it; ++it) {
                        const Index col = it.col();
                        const Real aij = it.value();

                        if (col < i) {
                            // Lower triangular part: use updated x (Gauss-Seidel)
                            sum -= aij * x(col);
                        }
                        else if (col > i) {
                            // Upper triangular part: use old x
                            sum -= aij * x(col);
                        }
                        // col == i: diagonal, skip - we use pre-computed diagonal_ instead
                    }

                    x(i) = sum / diagonal_(i);
                }
            }
        }

    private:
        int num_sweeps_ = 1;
        Vector diagonal_;
    };

} // namespace mpfem

#endif // MPFEM_GAUSS_SEIDEL_OPERATOR_HPP
#ifndef MPFEM_SPARSE_LU_OPERATOR_HPP
#define MPFEM_SPARSE_LU_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <Eigen/UmfPackSupport>
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Sparse LU direct solver operator.
     *
     * General-purpose sparse LU factorization for square matrices.
     * setup() performs symbolic analysis + factorization (expensive).
     * apply() does forward/back substitution (fast).
     * Caches factorization and reuses when matrix timestamp is unchanged.
     */
    class SparseLUOperator : public LinearOperator {
    public:
        SparseLUOperator() { operator_name_ = "SparseLU"; }

        void set_parameters(const ParameterList& params) override
        {
            (void)params;
        }

        void setup(const SparseMatrix* A) override
        {
            // Skip if matrix timestamp unchanged (cache hit) - O(1) check
            if (is_setup_ && A->timestamp() == lastMatrixTimestamp_) {
                return; // Reuse existing factorization
            }

            A_ = A;
            solver_.compute(A->eigen());

            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("SparseLUOperator: factorization failed");
            }

            lastMatrixTimestamp_ = A->timestamp();
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("SparseLUOperator: not setup");
            }

            x = solver_.solve(b);

            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("SparseLUOperator: solve failed");
            }
        }

    private:
        Eigen::SparseLU<SparseMatrix::Storage, Eigen::COLAMDOrdering<int>> solver_;
        std::uint64_t lastMatrixTimestamp_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_SPARSE_LU_OPERATOR_HPP

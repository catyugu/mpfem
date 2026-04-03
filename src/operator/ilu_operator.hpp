#ifndef MPFEM_ILU_OPERATOR_HPP
#define MPFEM_ILU_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace mpfem {

    /**
     * @brief Incomplete LU (ILU) operator with fill-in.
     *
     * General-purpose incomplete factorization for non-SPD matrices.
     * Uses Eigen::IncompleteLUT with configurable fill level and drop tolerance.
     *
     * Can be used as a preconditioner or standalone solver.
     */
    class ILUOperator : public LinearOperator {
    public:
        ILUOperator()
        {
            operator_name_ = "ILU";
        }

        void set_parameters(const ParameterList& params) override
        {
            if (params.has("FillLevel")) {
                fill_level_ = params.get_int("FillLevel");
            }
            if (params.has("DropTolerance")) {
                drop_tolerance_ = params.get_double("DropTolerance");
            }
        }

        void setup(const SparseMatrix* A) override
        {
            A_ = A;
            precond_.setDroptol(drop_tolerance_);
            precond_.setFillfactor(fill_level_);
            precond_.compute(A->eigen());
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_) {
                throw std::runtime_error("ILUOperator: setup() must be called before apply()");
            }
            x = precond_.solve(b);
        }

    private:
        Eigen::IncompleteLUT<Real> precond_;
        Real drop_tolerance_ = 1e-4;
        int fill_level_ = 10;
    };

} // namespace mpfem

#endif // MPFEM_ILU_OPERATOR_HPP
#include "solver/eigen_solver.hpp"

namespace mpfem {

    void EigenSparseLUOperator::setup(const SparseMatrix* A)
    {
        if (!A)
            throw std::runtime_error("EigenSparseLUOperator: null matrix in setup");
        solver_.compute(A->eigen());
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("EigenSparseLUOperator: factorization failed");
        }
        set_matrix(A);
        mark_setup();
    }

    void EigenSparseLUOperator::apply(const Vector& b, Vector& x)
    {
        x = solver_.solve(b);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("EigenSparseLUOperator: solve failed");
        }
    }

} // namespace mpfem
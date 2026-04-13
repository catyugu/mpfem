#ifndef MPFEM_EIGEN_PRECONDITIONER_ADAPTER_HPP
#define MPFEM_EIGEN_PRECONDITIONER_ADAPTER_HPP

#include "solver/linear_operator.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace mpfem {

    /**
     * @brief Bridge between Eigen solvers and our LinearOperator.
     * Enables any LinearOperator to be used as a preconditioner for Eigen IterativeSolvers.
     */
    class EigenPreconditionerAdapter {
    public:
        using MatrixType = Eigen::SparseMatrix<Real>;
        using Scalar = Real;
        using StorageIndex = int;
        using Index = Eigen::Index;

        enum {
            ColsAtCompileTime = Eigen::Dynamic,
            RowsAtCompileTime = Eigen::Dynamic,
            MaxColsAtCompileTime = Eigen::Dynamic,
            MaxRowsAtCompileTime = Eigen::Dynamic
        };

        EigenPreconditionerAdapter() : op_(nullptr), dim_(0) { }

        // Eigen Preconditioner Concept requirements
        EigenPreconditionerAdapter& analyzePattern(const MatrixType& A)
        {
            dim_ = A.rows();
            return *this;
        }
        EigenPreconditionerAdapter& factorize(const MatrixType& A)
        {
            dim_ = A.rows();
            return *this;
        }
        EigenPreconditionerAdapter& compute(const MatrixType& A)
        {
            dim_ = A.rows();
            return *this;
        }
        Eigen::ComputationInfo info() const { return Eigen::Success; }

        Index rows() const { return dim_; }
        Index cols() const { return dim_; }

        void set_operator(LinearOperator* op) { op_ = op; }

        /**
         * @brief Implementation of the solve operation required by Eigen.
         */
        template <typename Rhs, typename Dest>
        void _solve_impl(const Rhs& b, Dest& x) const
        {
            if (op_) {
                Vector b_vec = b;
                Vector x_vec(b.rows());
                op_->apply(b_vec, x_vec);
                x = x_vec;
            }
            else {
                x = b;
            }
        }

        template <typename Rhs>
        inline const Eigen::Solve<EigenPreconditionerAdapter, Rhs>
        solve(const Eigen::MatrixBase<Rhs>& b) const
        {
            return Eigen::Solve<EigenPreconditionerAdapter, Rhs>(*this, b.derived());
        }

    private:
        LinearOperator* op_;
        Index dim_;
    };

} // namespace mpfem

// Inject Eigen traits for the custom preconditioner
namespace Eigen {
    namespace internal {

        template <>
        struct traits<mpfem::EigenPreconditionerAdapter> : traits<Eigen::SparseMatrix<mpfem::Real>> { };

        template <typename Rhs>
        struct traits<Solve<mpfem::EigenPreconditionerAdapter, Rhs>> : traits<typename Rhs::PlainObject> {
            typedef typename Rhs::PlainObject PlainObject;
            typedef typename PlainObject::StorageKind StorageKind;
        };

        template <typename Rhs>
        struct evaluator<Solve<mpfem::EigenPreconditionerAdapter, Rhs>>
            : evaluator<typename Rhs::PlainObject> {
            using PlainObject = typename Rhs::PlainObject;
            using Base = evaluator<PlainObject>;

            evaluator(const Solve<mpfem::EigenPreconditionerAdapter, Rhs>& solve)
                : m_result(solve.rows(), solve.cols())
            {
                solve.dec()._solve_impl(solve.rhs(), m_result);
                ::new (static_cast<Base*>(this)) Base(m_result);
            }

        protected:
            PlainObject m_result;
        };
    } // namespace internal
} // namespace Eigen

#endif // MPFEM_EIGEN_PRECONDITIONER_ADAPTER_HPP

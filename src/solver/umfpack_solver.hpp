#ifndef MPFEM_UMFPACK_SOLVER_HPP
#define MPFEM_UMFPACK_SOLVER_HPP

#include "linear_operator.hpp"
#include <memory>

namespace mpfem {

    /**
     * @brief SuiteSparse UMFPACK direct solver.
     *
     * High-performance direct LU solver from SuiteSparse.
     * Good alternative when MKL PARDISO is not available.
     */
    class UmfpackSolver : public LinearOperator {
    public:
        UmfpackSolver();
        ~UmfpackSolver() override;

        std::string_view name() const override { return "UMFPACK"; }

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;

        int iterations() const override;
        Real residual() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace mpfem

#endif // MPFEM_UMFPACK_SOLVER_HPP

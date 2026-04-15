#ifndef MPFEM_PARDISO_SOLVER_HPP
#define MPFEM_PARDISO_SOLVER_HPP

#include "linear_operator.hpp"
#include <memory>

namespace mpfem {

    /**
     * @brief MKL PARDISO direct solver.
     *
     * High-performance direct solver from Intel MKL.
     * Supports both symmetric and unsymmetric matrices.
     */
    class PardisoSolver : public LinearOperator {
    public:
        PardisoSolver();
        ~PardisoSolver() override;

        std::string_view name() const override { return "Pardiso"; }

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;

        int iterations() const override;
        Real residual() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace mpfem

#endif // MPFEM_PARDISO_SOLVER_HPP

#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "linear_operator.hpp"
#include <memory>

namespace mpfem {

    /**
     * @brief Conjugate Gradient solver for symmetric positive definite matrices.
     */
    class CgOperator : public LinearOperator {
    public:
        CgOperator();
        ~CgOperator() override;

        std::string_view name() const override { return "CG"; }

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;
        void configure(const LinearOperatorConfig& config) override;

        int iterations() const override;
        Real residual() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /**
     * @brief Dynamic GMRES solver for general unsymmetric matrices.
     */
    class GmresOperator : public LinearOperator {
    public:
        GmresOperator();
        ~GmresOperator() override;

        std::string_view name() const override { return "DGMRES"; }

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;
        void configure(const LinearOperatorConfig& config) override;

        int iterations() const override;
        Real residual() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /**
     * @brief Direct Solver: Eigen SparseLU
     */
    class EigenSparseLUOperator : public LinearOperator {
    public:
        EigenSparseLUOperator();
        ~EigenSparseLUOperator() override;

        std::string_view name() const override { return "SparseLU"; }

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace mpfem

#endif // MPFEM_EIGEN_SOLVER_HPP

#ifndef MPFEM_PARDISO_SOLVER_HPP
#define MPFEM_PARDISO_SOLVER_HPP

#include "core/logger.hpp"
#include "linear_operator.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef MPFEM_USE_MKL
#include <Eigen/Sparse>
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#endif

namespace mpfem {

#ifdef MPFEM_USE_MKL

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

        std::string_view name() const override;

        void setup(const SparseMatrix* A) override;
        void apply(const Vector& b, Vector& x) override;

        int iterations() const override;
        Real residual() const override;

    private:
        void convertToCSR(const SparseMatrix::Storage& mat);

        // PARDISO handle array (64 pointers)
        _MKL_DSS_HANDLE_t pt_[64];

        // Control parameters
        MKL_INT iparm_[64] = {0};

        // CSR format (1-based indexing)
        std::vector<MKL_INT> ia_; // Row pointers (n+1 elements)
        std::vector<MKL_INT> ja_; // Column indices (nnz elements)
        std::vector<Real> a_; // Values (nnz elements)

        MKL_INT n_ = 0;
        MKL_INT maxfct_ = 1;
        MKL_INT mnum_ = 1;
        MKL_INT mtype_ = 11; // Real unsymmetric
        MKL_INT phase_ = 0;
        MKL_INT msglvl_ = 0;
        MKL_INT error_ = 0;

        bool initialized_ = false;
        int iterations_ = 1;
        Real residual_ = 0.0;
    };

#else

    // Stub for when MKL is not available
    class PardisoSolver : public LinearOperator {
    public:
        PardisoSolver()
        {
            throw std::runtime_error("PardisoSolver: MKL not available. "
                                     "Rebuild with MKL support enabled.");
        }
        std::string_view name() const override { return "Pardiso"; }
        void setup(const SparseMatrix*) override
        {
            throw std::runtime_error("PardisoSolver: MKL not available");
        }
        void apply(const Vector&, Vector&) override
        {
            throw std::runtime_error("PardisoSolver: MKL not available");
        }
    };

#endif // MPFEM_USE_MKL

} // namespace mpfem

#endif // MPFEM_PARDISO_SOLVER_HPP
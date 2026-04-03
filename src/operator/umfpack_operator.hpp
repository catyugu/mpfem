#ifndef MPFEM_UMFPACK_OPERATOR_HPP
#define MPFEM_UMFPACK_OPERATOR_HPP

#include "operator/linear_operator.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <umfpack.h>

namespace mpfem {

    /**
     * @brief UMFPACK direct solver operator using SuiteSparse directly.
     *
     * Uses UMFPACK (Unsymmetric MultiFrontal Package) for sparse LU decomposition.
     * This is a high-performance direct solver for sparse systems.
     *
     * - setup(A): Symbolic analysis + numerical factorization (expensive)
     * - apply(b, x): Forward/back substitution (fast)
     * - Caches factorization for reuse when matrix pattern is unchanged
     */
    class UMFPACKOperator : public LinearOperator {
    public:
        UMFPACKOperator() { operator_name_ = "UMFPACK"; }

        ~UMFPACKOperator() { freeFactorization(); }

        void set_parameters(const ParameterList& params) override { (void)params; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("UMFPACKOperator: null matrix");
            }

            // Check if we can reuse previous factorization
            std::uint64_t currentFingerprint = A->fingerprint();
            if (currentFingerprint == lastMatrixFingerprint_ && symbolic_ != nullptr) {
                // Reuse existing factorization - just need numeric step
                numericReFactorize(A);
                is_setup_ = true;
                return;
            }

            // Need full setup: symbolic + numeric
            freeFactorization();
            A_ = A;

            // Get matrix dimensions - UMFPACK uses int64_t
            int64_t n = static_cast<int64_t>(A->eigen().rows());
            int64_t nnz = static_cast<int64_t>(A->eigen().nonZeros());

            const auto& eigenA = A->eigen();

            // Convert to CSC format (UMFPACK requirement) with int64_t
            ap_.resize(n + 1);
            ai_.resize(nnz);
            ax_.resize(nnz);
            convertCSRtoCSC64(eigenA, ap_.data(), ai_.data(), ax_.data());

            // Symbolic factorization
            void* symbolic = nullptr;
            int status = umfpack_dl_symbolic(n, n, ap_.data(), ai_.data(), ax_.data(),
                &symbolic, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                throw std::runtime_error("UMFPACKOperator: symbolic factorization failed");
            }
            symbolic_ = symbolic;

            // Numeric factorization
            void* numeric = nullptr;
            status = umfpack_dl_numeric(ap_.data(), ai_.data(), ax_.data(),
                symbolic, &numeric, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                umfpack_dl_free_symbolic(&symbolic);
                throw std::runtime_error("UMFPACKOperator: numeric factorization failed");
            }
            numeric_ = numeric;

            lastMatrixFingerprint_ = currentFingerprint;
            is_setup_ = true;
        }

        void apply(const Vector& b, Vector& x) override
        {
            if (!is_setup_ || !numeric_) {
                throw std::runtime_error("UMFPACKOperator: not setup");
            }

            // Only resize if needed (avoid unnecessary allocations in hot path)
            if (x.size() != static_cast<Index>(b.size())) {
                x.resize(b.size());
            }

            int status = umfpack_dl_solve(UMFPACK_A,
                ap_.data(), ai_.data(), ax_.data(),
                x.data(), b.data(),
                numeric_, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                throw std::runtime_error("UMFPACKOperator: solve failed");
            }
        }

        void apply_transpose(const Vector& b, Vector& x) override
        {
            if (!is_setup_ || !numeric_) {
                throw std::runtime_error("UMFPACKOperator: not setup");
            }

            // Only resize if needed (avoid unnecessary allocations in hot path)
            if (x.size() != static_cast<Index>(b.size())) {
                x.resize(b.size());
            }

            // Solve A^T x = b
            int status = umfpack_dl_solve(UMFPACK_Aat,
                ap_.data(), ai_.data(), ax_.data(),
                x.data(), b.data(),
                numeric_, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                throw std::runtime_error("UMFPACKOperator: transpose solve failed");
            }
        }

    private:
        void* symbolic_ = nullptr;
        void* numeric_ = nullptr;
        std::uint64_t lastMatrixFingerprint_ = 0;
        std::vector<int64_t> ap_; // CSC row pointer (int64_t for UMFPACK)
        std::vector<int64_t> ai_; // CSC column indices
        std::vector<double> ax_; // CSC values

        void freeFactorization()
        {
            if (symbolic_) {
                umfpack_dl_free_symbolic(&symbolic_);
                symbolic_ = nullptr;
            }
            if (numeric_) {
                umfpack_dl_free_numeric(&numeric_);
                numeric_ = nullptr;
            }
            lastMatrixFingerprint_ = 0;
            is_setup_ = false;
        }

        void numericReFactorize(const SparseMatrix* A)
        {
            if (symbolic_) {
                umfpack_dl_free_numeric(&numeric_);
                numeric_ = nullptr;
            }

            const auto& eigenA = A->eigen();
            int64_t n = static_cast<int64_t>(eigenA.rows());
            int64_t nnz = static_cast<int64_t>(eigenA.nonZeros());

            ap_.resize(n + 1);
            ai_.resize(nnz);
            ax_.resize(nnz);
            convertCSRtoCSC64(eigenA, ap_.data(), ai_.data(), ax_.data());

            void* numeric = nullptr;
            int status = umfpack_dl_numeric(ap_.data(), ai_.data(), ax_.data(),
                symbolic_, &numeric, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                throw std::runtime_error("UMFPACKOperator: numeric re-factorization failed");
            }
            numeric_ = numeric;
        }

        void convertCSRtoCSC64(const Eigen::SparseMatrix<double>& csr,
            int64_t* ap, int64_t* ai, double* ax)
        {
            int64_t n = csr.rows();
            int64_t nnz = csr.nonZeros();

            const int* csr_row_ptr = csr.outerIndexPtr();
            const int* csr_col_idx = csr.innerIndexPtr();
            const double* csr_vals = csr.valuePtr();

            // Count nnz per column for CSC
            std::vector<int64_t> col_nnz(n, 0);
            for (int64_t i = 0; i < nnz; ++i) {
                col_nnz[csr_col_idx[i]]++;
            }

            // Build CSC row pointer array
            ap[0] = 0;
            for (int64_t j = 0; j < n; ++j) {
                ap[j + 1] = ap[j] + col_nnz[j];
            }

            // Fill CSC arrays
            std::vector<int64_t> current_pos(n, 0);
            for (int64_t i = 0; i < n; ++i) {
                for (int idx = csr_row_ptr[i]; idx < csr_row_ptr[i + 1]; ++idx) {
                    int64_t j = csr_col_idx[idx];
                    int64_t pos = ap[j] + current_pos[j]++;
                    ai[pos] = i;
                    ax[pos] = csr_vals[idx];
                }
            }
        }
    };

} // namespace mpfem

#endif // MPFEM_UMFPACK_OPERATOR_HPP

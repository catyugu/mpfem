#ifndef MPFEM_PARDISO_SOLVER_HPP
#define MPFEM_PARDISO_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>

#ifdef MPFEM_USE_MKL
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#include <Eigen/Sparse>
#endif

namespace mpfem {

#ifdef MPFEM_USE_MKL

/**
 * @brief MKL PARDISO direct solver.
 * 
 * High-performance direct solver from Intel MKL.
 * Supports both symmetric and unsymmetric matrices.
 */
class PardisoSolver : public LinearSolver {
public:
    PardisoSolver() {
        // Initialize handle array to null
        std::memset(pt_, 0, sizeof(pt_));
        
        // Initialize iparm with default values using pardisoinit
        mtype_ = 11;  // Real unsymmetric matrix
        pardisoinit(pt_, &mtype_, iparm_);
        
        // Override some default parameters
        iparm_[0] = 1;   // Use user-defined iparm values
        iparm_[1] = 2;   // Fill-in reducing: METIS
        // Note: iparm_[34] = 0 (default) means 1-based indexing for ia/ja arrays
        
        maxfct_ = 1;
        mnum_ = 1;
        phase_ = 0;
        msglvl_ = 0;  // No output
        error_ = 0;
    }
    
    ~PardisoSolver() override {
        // Release all memory
        if (initialized_) {
            phase_ = -1;
            MKL_INT nrhs = 1;
            MKL_INT dummy_n = 0;
            pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                    &dummy_n, nullptr, nullptr, nullptr,
                    nullptr, &nrhs, iparm_, &msglvl_, nullptr, nullptr, &error_);
        }
    }
    
    std::string name() const override { return "mkl.pardiso"; }
    
    void setPrintLevel(int level) override {
        printLevel_ = level;
        msglvl_ = (level >= 2) ? 1 : 0;
    }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (PARDISO)");
        
        const auto& mat = A.eigen();
        n_ = static_cast<MKL_INT>(mat.rows());
        const std::uint64_t currentFingerprint = A.fingerprint();
        const bool needRefactor = !hasFactorCache_ || (currentFingerprint != lastMatrixFingerprint_);

        if (needRefactor) {
            convertToCSR(mat);
        }
        
        MKL_INT nrhs = 1;

        if (needRefactor) {
            // Analysis + factorization + solve
            phase_ = 13;
        } else {
            // Solve only with cached factors
            phase_ = 33;
            if (printLevel_ >= 1) {
                LOG_INFO << "[PARDISO] Reusing cached factorization";
            }
        }
        
        initialized_ = true;
        
        x.resize(n_);
        
        // PARDISO modifies RHS, so copy to mutable buffer
        Vector b_copy = b;
        
        // Solve
        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                &n_, a_.data(), ia_.data(), ja_.data(),
                nullptr, &nrhs, iparm_, &msglvl_, 
                b_copy.data(), x.data(), &error_);
        
        if (error_ != 0) {
            LOG_ERROR << "PARDISO solve failed with error: " << error_;
            hasFactorCache_ = false;
            return false;
        }
        
        hasFactorCache_ = true;
        lastMatrixFingerprint_ = currentFingerprint;
        
        iterations_ = 1;
        residual_ = 0.0;
        
        LOG_INFO << "[PARDISO] Solve successful, solution norm: " << x.norm();
        return true;
    }

private:
    void convertToCSR(const SparseMatrix::Storage& mat) {
        const MKL_INT n = static_cast<MKL_INT>(mat.rows());
        const MKL_INT nnz = static_cast<MKL_INT>(mat.nonZeros());
        
        // PARDISO requires CSR format with 1-based indexing
        ia_.resize(n + 1);
        ja_.resize(nnz);
        a_.resize(nnz);
        
        // Convert from column-major to row-major CSR format
        // Count entries per row first
        std::vector<MKL_INT> rowCounts(n, 0);
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
            for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                rowCounts[it.row()]++;
            }
        }
        
        // Build row pointers (1-based)
        ia_[0] = 1;
        for (MKL_INT i = 0; i < n; ++i) {
            ia_[i + 1] = ia_[i] + rowCounts[i];
        }
        
        // Fill column indices and values
        std::vector<MKL_INT> rowOffsets(n, 0);
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(mat.cols()); ++j) {
            for (SparseMatrix::Storage::InnerIterator it(mat, j); it; ++it) {
                MKL_INT row = static_cast<MKL_INT>(it.row());
                MKL_INT pos = ia_[row] - 1 + rowOffsets[row];  // 0-based position
                ja_[pos] = static_cast<MKL_INT>(j) + 1;  // 1-based column index
                a_[pos] = it.value();
                rowOffsets[row]++;
            }
        }
    }
    
    // PARDISO handle array (64 pointers)
    _MKL_DSS_HANDLE_t pt_[64];
    
    // Control parameters
    MKL_INT iparm_[64] = {0};
    
    // CSR format (1-based indexing)
    std::vector<MKL_INT> ia_;   // Row pointers (n+1 elements)
    std::vector<MKL_INT> ja_;   // Column indices (nnz elements)
    std::vector<Real> a_;       // Values (nnz elements)
    
    MKL_INT n_ = 0;
    MKL_INT maxfct_ = 1;
    MKL_INT mnum_ = 1;
    MKL_INT mtype_ = 11;  // Real unsymmetric
    MKL_INT phase_ = 0;
    MKL_INT msglvl_ = 0;
    MKL_INT error_ = 0;
    
    bool initialized_ = false;
    bool hasFactorCache_ = false;
    std::uint64_t lastMatrixFingerprint_ = 0;
};

#else

// Stub for when MKL is not available
class PardisoSolver : public LinearSolver {
public:
    PardisoSolver() {
        throw std::runtime_error("PardisoSolver: MKL not available. "
                                 "Rebuild with MKL support enabled.");
    }
    std::string name() const override { return "mkl.pardiso"; }
    bool solve(const SparseMatrix&, Vector&, const Vector&) override { return false; }
};

#endif  // MPFEM_USE_MKL

}  // namespace mpfem

#endif  // MPFEM_PARDISO_SOLVER_HPP

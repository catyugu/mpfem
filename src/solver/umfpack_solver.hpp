#ifndef MPFEM_UMFPACK_SOLVER_HPP
#define MPFEM_UMFPACK_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"

#ifdef MPFEM_USE_UMFPACK
#include <umfpack.h>
#include <Eigen/Sparse>
#endif

namespace mpfem {

#ifdef MPFEM_USE_UMFPACK

/**
 * @brief UMFPACK direct solver from SuiteSparse.
 * 
 * Unsymmetric Multifrontal Sparse LU Factorization Package.
 * High-performance direct solver for sparse linear systems.
 * 
 * Only available when MPFEM_USE_UMFPACK is defined.
 */
class UmfpackSolver : public LinearSolver {
public:
    UmfpackSolver() = default;
    
    ~UmfpackSolver() override {
        if (numeric_) {
            umfpack_di_free_numeric(&numeric_);
            numeric_ = nullptr;
        }
        if (symbolic_) {
            umfpack_di_free_symbolic(&symbolic_);
            symbolic_ = nullptr;
        }
    }
    
    std::string name() const override { return "umfpack.lu"; }
    
    void analyzePattern(const SparseMatrix& A) override {
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        
        // Convert to CSC format
        convertToCSC(mat);
        
        // Symbolic analysis
        int status = umfpack_di_symbolic(n_, n_, 
            ap_.data(), ai_.data(), ax_.data(), 
            &symbolic_, nullptr, nullptr);
        
        if (status != UMFPACK_OK) {
            LOG_ERROR << "UMFPACK symbolic analysis failed with status: " << status;
            return;
        }
        
        analyzed_ = true;
        LOG_DEBUG << "UMFPACK: Symbolic analysis completed";
    }
    
    void factorize(const SparseMatrix& A) override {
        if (!analyzed_) {
            analyzePattern(A);
        }
        
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        
        // Convert to CSC format
        convertToCSC(mat);
        
        // Free previous numeric factorization
        if (numeric_) {
            umfpack_di_free_numeric(&numeric_);
            numeric_ = nullptr;
        }
        
        // Numeric factorization
        int status = umfpack_di_numeric(
            ap_.data(), ai_.data(), ax_.data(),
            symbolic_, &numeric_, nullptr, nullptr);
        
        if (status != UMFPACK_OK) {
            LOG_ERROR << "UMFPACK numeric factorization failed with status: " << status;
            return;
        }
        
        factorized_ = true;
        LOG_DEBUG << "UMFPACK: Numeric factorization completed";
    }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (UMFPACK)");
        
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        
        // Convert to CSC format
        convertToCSC(mat);
        
        // If not factorized, do full solve
        if (!factorized_) {
            // Symbolic analysis
            if (!symbolic_) {
                int status = umfpack_di_symbolic(n_, n_, 
                    ap_.data(), ai_.data(), ax_.data(), 
                    &symbolic_, nullptr, nullptr);
                if (status != UMFPACK_OK) {
                    LOG_ERROR << "UMFPACK symbolic analysis failed";
                    return false;
                }
            }
            
            // Numeric factorization
            if (numeric_) {
                umfpack_di_free_numeric(&numeric_);
            }
            int status = umfpack_di_numeric(
                ap_.data(), ai_.data(), ax_.data(),
                symbolic_, &numeric_, nullptr, nullptr);
            if (status != UMFPACK_OK) {
                LOG_ERROR << "UMFPACK numeric factorization failed";
                return false;
            }
            factorized_ = true;
        }
        
        // Solve
        x.resize(n_);
        int status = umfpack_di_solve(UMFPACK_A,
            ap_.data(), ai_.data(), ax_.data(),
            x.data(), const_cast<Real*>(b.data()),
            numeric_, nullptr, nullptr);
        
        if (status != UMFPACK_OK) {
            LOG_ERROR << "UMFPACK solve failed with status: " << status;
            return false;
        }
        
        iterations_ = 1;
        residual_ = 0.0;
        
        LOG_INFO << "[UMFPACK] Solve successful, solution norm: " << x.norm();
        return true;
    }
    
private:
    void convertToCSC(const Eigen::SparseMatrix<Real, Eigen::RowMajor>& mat) {
        const int n = static_cast<int>(mat.rows());
        const int nnz = static_cast<int>(mat.nonZeros());
        
        // UMFPACK requires CSC format with 0-based indexing
        // Arrays: Ap (column pointers), Ai (row indices), Ax (values)
        
        ap_.resize(n + 1);
        ai_.resize(nnz);
        ax_.resize(nnz);
        
        // Count entries per column
        std::vector<int> colCount(n, 0);
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(mat, k); it; ++it) {
                colCount[it.col()]++;
            }
        }
        
        // Build column pointers
        ap_[0] = 0;
        for (int i = 0; i < n; ++i) {
            ap_[i + 1] = ap_[i] + colCount[i];
        }
        
        // Fill row indices and values
        std::vector<int> colPos = ap_;
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(mat, k); it; ++it) {
                const int col = it.col();
                const int pos = colPos[col];
                ai_[pos] = it.row();
                ax_[pos] = it.value();
                colPos[col]++;
            }
        }
    }
    
    // UMFPACK data structures
    void* symbolic_ = nullptr;
    void* numeric_ = nullptr;
    
    // CSC format (0-based indexing)
    std::vector<int> ap_;   // Column pointers (n+1 elements)
    std::vector<int> ai_;   // Row indices (nnz elements)
    std::vector<Real> ax_;  // Values (nnz elements)
    
    int n_ = 0;
    bool analyzed_ = false;
    bool factorized_ = false;
};

#else

// Stub for when UMFPACK is not available
class UmfpackSolver : public LinearSolver {
public:
    UmfpackSolver() {
        throw std::runtime_error("UmfpackSolver: UMFPACK not available. "
                                 "Rebuild with -DMPFEM_USE_UMFPACK=ON");
    }
    std::string name() const override { return "umfpack.lu"; }
    bool solve(const SparseMatrix&, Vector&, const Vector&) override { return false; }
};

#endif  // MPFEM_USE_UMFPACK

}  // namespace mpfem

#endif  // MPFEM_UMFPACK_SOLVER_HPP

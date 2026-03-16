#ifndef MPFEM_PARDISO_SOLVER_HPP
#define MPFEM_PARDISO_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"

#ifdef MPFEM_USE_MKL
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#include <Eigen/Sparse>
#endif

namespace mpfem {

#ifdef MPFEM_USE_MKL

/**
 * @brief Intel MKL PARDISO direct solver.
 * 
 * High-performance direct solver for sparse linear systems.
 * Supports both symmetric and non-symmetric matrices.
 * 
 * Only available when MPFEM_USE_MKL is defined.
 */
class PardisoSolver : public LinearSolver {
public:
    PardisoSolver() {
        // Initialize internal address pointer
        for (int i = 0; i < 64; ++i) {
            pt_[i] = 0;
        }
        
        // Default parameters
        // iparm[0] = 1 means use user-defined iparm values
        iparm_[0] = 1;
        
        // Use recommended default values
        iparm_[1] = 2;    // Fill-in reducing ordering: 2=METIS
        iparm_[2] = 0;    // Reserved
        iparm_[3] = 0;    // No iterative-direct algorithm
        iparm_[4] = 0;    // No user fill-in reducing permutation
        iparm_[5] = 0;    // Write solution into x
        iparm_[6] = 0;    // Output: number of iterative refinement steps
        iparm_[7] = 2;    // Maximum number of iterative refinement steps
        iparm_[8] = 0;    // Reserved
        iparm_[9] = 13;   // Perturb the pivot elements with 1E-13
        iparm_[10] = 1;   // Use nonsymmetric permutation and scaling MPS
        iparm_[11] = 0;   // Solve with A^T x = b
        iparm_[12] = 1;   // Maximum weighted matching algorithm
        iparm_[13] = 0;   // Output: number of perturbed pivots
        iparm_[14] = 0;   // Output: memory usage in peak
        iparm_[15] = 0;   // Output: memory usage for factorization
        iparm_[16] = 0;   // Output: memory usage for solution
        iparm_[17] = 0;   // Output: number of non-zero in LU factors
        iparm_[18] = 0;   // Output: number of floating-point operations
        iparm_[19] = 0;   // Output: CG/CGS diagnostics
        iparm_[20] = 1;   // Pivoting for symmetric indefinite matrices
        iparm_[26] = 0;   // Matrix checking: 0=off, 1=on
        iparm_[27] = 0;   // Double precision: 0=double, 1=single
        iparm_[34] = 0;   // One-based indexing: 0=C-style (0-based)
        iparm_[59] = 0;   // OOC mode: 0=in-core
        
        // Initialize error flag
        error_ = 0;
    }
    
    ~PardisoSolver() override {
        // Release memory
        if (initialized_) {
            phase_ = -1;
            PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                    &n_, nullptr, nullptr, nullptr, nullptr, &nrhs_,
                    iparm_, &msglvl_, nullptr, nullptr, &error_);
        }
    }
    
    std::string name() const override { return "mkl.pardiso"; }
    
    void analyzePattern(const SparseMatrix& A) override {
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        nnz_ = static_cast<int>(mat.nonZeros());
        
        // Convert to CSC format required by PARDISO
        convertToCSC(mat);
        
        // Phase 1: Reordering and Symbolic Factorization
        phase_ = 11;
        PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                &n_, valuesCSC_.data(), rowIndexCSC_.data(), colsIndexCSC_.data(),
                nullptr, &nrhs_, iparm_, &msglvl_, nullptr, nullptr, &error_);
        
        if (error_ != 0) {
            LOG_ERROR << "PARDISO analysis failed with error: " << error_;
            return;
        }
        
        analyzed_ = true;
        LOG_DEBUG << "PARDISO: Analysis completed";
    }
    
    void factorize(const SparseMatrix& A) override {
        if (!analyzed_) {
            analyzePattern(A);
        }
        
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        nnz_ = static_cast<int>(mat.nonZeros());
        
        // Update values (CSC format stays the same for same sparsity pattern)
        convertToCSC(mat);
        
        // Phase 2: Numerical Factorization
        phase_ = 22;
        PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                &n_, valuesCSC_.data(), rowIndexCSC_.data(), colsIndexCSC_.data(),
                nullptr, &nrhs_, iparm_, &msglvl_, nullptr, nullptr, &error_);
        
        if (error_ != 0) {
            LOG_ERROR << "PARDISO factorization failed with error: " << error_;
            return;
        }
        
        factorized_ = true;
        LOG_DEBUG << "PARDISO: Factorization completed";
    }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (PARDISO)");
        
        const auto& mat = A.eigen();
        n_ = static_cast<int>(mat.rows());
        nnz_ = static_cast<int>(mat.nonZeros());
        
        // Convert to CSC format
        convertToCSC(mat);
        
        if (!factorized_) {
            // Full solve: analysis + factorization + solve
            phase_ = 13;
        } else {
            // Solve only (using existing factorization)
            phase_ = 33;
        }
        
        // Solve
        x.resize(n_);
        PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase_,
                &n_, valuesCSC_.data(), rowIndexCSC_.data(), colsIndexCSC_.data(),
                nullptr, &nrhs_, iparm_, &msglvl_, 
                const_cast<Real*>(b.data()), x.data(), &error_);
        
        if (error_ != 0) {
            LOG_ERROR << "PARDISO solve failed with error: " << error_;
            return false;
        }
        
        initialized_ = true;
        factorized_ = true;
        iterations_ = 1;
        residual_ = 0.0;
        
        LOG_INFO << "[PARDISO] Solve successful, solution norm: " << x.norm();
        return true;
    }
    
private:
    void convertToCSC(const Eigen::SparseMatrix<Real, Eigen::RowMajor>& mat) {
        // PARDISO requires CSC format with 1-based indexing
        // Convert from Eigen's row-major CSR to CSC
        
        const int n = static_cast<int>(mat.rows());
        const int nnz = static_cast<int>(mat.nonZeros());
        
        // Resize arrays
        valuesCSC_.resize(nnz);
        rowIndexCSC_.resize(nnz);
        colsIndexCSC_.resize(n + 1);
        
        // Create column pointers (CSC)
        // First count entries per column
        std::vector<int> colCount(n, 0);
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(mat, k); it; ++it) {
                colCount[it.col()]++;
            }
        }
        
        // Convert to cumulative counts (column pointers)
        colsIndexCSC_[0] = 1;  // 1-based
        for (int i = 0; i < n; ++i) {
            colsIndexCSC_[i + 1] = colsIndexCSC_[i] + colCount[i];
        }
        
        // Fill values and row indices
        std::vector<int> colPos = colsIndexCSC_;  // Current position in each column
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Real, Eigen::RowMajor>::InnerIterator it(mat, k); it; ++it) {
                const int col = it.col();
                const int pos = colPos[col] - 1;  // Convert to 0-based for array access
                valuesCSC_[pos] = it.value();
                rowIndexCSC_[pos] = it.row() + 1;  // 1-based row index
                colPos[col]++;
            }
        }
    }
    
    // PARDISO internal data
    void* pt_[64] = {};
    int iparm_[64] = {};
    int maxfct_ = 1;    // Maximum number of factorizations
    int mnum_ = 1;      // Matrix number
    int mtype_ = 11;    // Matrix type: 11=real unsymmetric
    int phase_ = 0;     // Phase flag
    int n_ = 0;         // Number of equations
    int nnz_ = 0;       // Number of non-zeros
    int nrhs_ = 1;      // Number of right-hand sides
    int msglvl_ = 0;    // Message level: 0=quiet
    int error_ = 0;     // Error flag
    
    // CSC format storage (1-based indexing for PARDISO)
    std::vector<Real> valuesCSC_;
    std::vector<int> rowIndexCSC_;
    std::vector<int> colsIndexCSC_;
    
    bool analyzed_ = false;
    bool factorized_ = false;
    bool initialized_ = false;
};

#else

// Stub for when MKL is not available
class PardisoSolver : public LinearSolver {
public:
    PardisoSolver() {
        throw std::runtime_error("PardisoSolver: MKL not available. "
                                 "Rebuild with -DMPFEM_USE_MKL=ON");
    }
    std::string name() const override { return "mkl.pardiso"; }
    bool solve(const SparseMatrix&, Vector&, const Vector&) override { return false; }
};

#endif  // MPFEM_USE_MKL

}  // namespace mpfem

#endif  // MPFEM_PARDISO_SOLVER_HPP

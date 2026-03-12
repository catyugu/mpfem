#ifndef MPFEM_PARDISO_SOLVER_HPP
#define MPFEM_PARDISO_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"

#ifdef MPFEM_USE_MKL
#include <mkl_pardiso.h>
#include <mkl_spblas.h>
#endif

namespace mpfem {

/**
 * @brief Intel MKL PARDISO direct solver.
 * 
 * High-performance sparse direct solver for large-scale problems.
 * Only available when MPFEM_USE_MKL is defined.
 */
class PardisoSolver : public LinearSolver {
public:
    PardisoSolver();
    ~PardisoSolver() override;
    
    std::string name() const override { return "MKL::PARDISO"; }
    
    void analyzePattern(const SparseMatrix& A) override;
    void factorize(const SparseMatrix& A) override;
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override;
    
    /// Set matrix type (0=auto, 1=real symmetric indefinite, 2=real SPD, 11=real unsymmetric)
    void setMatrixType(int type) { matrixType_ = type; }
    
private:
#ifdef MPFEM_USE_MKL
    void initPardiso();
    void freePardiso();
    void checkError(int error, const char* phase);
    
    // PARDISO internal data
    void* pt_[64] = {nullptr};
    int iparm_[64] = {0};
    int mtype_ = 11;        // Real unsymmetric matrix
    int nrhs_ = 1;
    int maxfct_ = 1;
    int mnum_ = 1;
    
    // Store matrix in CSR format for PARDISO
    std::vector<Real> a_;
    std::vector<int> ia_;
    std::vector<int> ja_;
    int n_ = 0;
    
    bool initialized_ = false;
    bool factorized_ = false;
#endif
    int matrixType_ = 0;  // Auto-detect
};

// Stub implementation when MKL is not available
#ifndef MPFEM_USE_MKL

inline PardisoSolver::PardisoSolver() {
    LOG_ERROR("PardisoSolver: MKL not available. Please compile with MPFEM_USE_MKL=ON");
}

inline PardisoSolver::~PardisoSolver() = default;

inline void PardisoSolver::analyzePattern(const SparseMatrix&) {
    LOG_ERROR("PardisoSolver: MKL not available");
}

inline void PardisoSolver::factorize(const SparseMatrix&) {
    LOG_ERROR("PardisoSolver: MKL not available");
}

inline bool PardisoSolver::solve(const SparseMatrix&, Vector&, const Vector&) {
    LOG_ERROR("PardisoSolver: MKL not available");
    return false;
}

#else

inline PardisoSolver::PardisoSolver() {
    initPardiso();
}

inline PardisoSolver::~PardisoSolver() {
    freePardiso();
}

inline void PardisoSolver::initPardiso() {
    // Initialize iparm
    iparm_[0] = 1;      // No default values
    iparm_[1] = 2;      // Fill-in reducing ordering: METIS
    iparm_[3] = 0;      // No iterative-direct algorithm
    iparm_[4] = 0;      // No user fill-in reducing permutation
    iparm_[5] = 0;      // Write solution into x
    iparm_[6] = 0;      // Output: Number of iterative refinement steps
    iparm_[7] = 2;      // Maximum number of iterative refinement steps
    iparm_[9] = 13;     // Perturb the pivot elements
    iparm_[10] = 1;     // Use nonsymmetric permutation and scaling
    iparm_[11] = 0;     // Solve with transposed or conjugate transposed matrix
    iparm_[12] = 1;     // Maximum weighted matching algorithm
    iparm_[13] = 0;     // Output: Number of perturbed pivots
    iparm_[17] = -1;    // Output: Number of nonzeros in LU factor
    iparm_[18] = -1;    // Output: MFlops for LU factorization
    iparm_[19] = 0;     // Output: Numbers of CG Iterations
    iparm_[26] = 0;     // Check matrix for errors
    iparm_[34] = 1;     // Use zero-based indexing (C-style)
    iparm_[59] = 0;     // Use in-core PARDISO
    
    initialized_ = true;
    LOG_DEBUG("PardisoSolver initialized");
}

inline void PardisoSolver::freePardiso() {
    if (!initialized_) return;
    
    int phase = -1;  // Release all memory
    int error = 0;
    Real dummy = 0.0;
    
    PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase,
            &n_, &dummy, ia_.data(), ja_.data(), 
            nullptr, &nrhs_, iparm_, &error, 
            nullptr, nullptr);
    
    if (error != 0) {
        LOG_ERROR("PARDISO release error: " << error);
    }
    
    initialized_ = false;
    factorized_ = false;
    LOG_DEBUG("PardisoSolver freed");
}

inline void PardisoSolver::analyzePattern(const SparseMatrix& A) {
    if (!initialized_) {
        initPardiso();
    }
    
    n_ = static_cast<int>(A.rows());
    
    // Convert Eigen sparse matrix to CSR format (0-based)
    const auto& eigen = A.eigen();
    eigen.makeCompressed();
    
    a_.resize(eigen.nonZeros());
    ia_.resize(n_ + 1);
    ja_.resize(eigen.nonZeros());
    
    // Copy data
    std::copy(eigen.valuePtr(), eigen.valuePtr() + eigen.nonZeros(), a_.begin());
    std::copy(eigen.outerIndexPtr(), eigen.outerIndexPtr() + n_ + 1, ia_.begin());
    std::copy(eigen.innerIndexPtr(), eigen.innerIndexPtr() + eigen.nonZeros(), ja_.begin());
    
    // Phase 11: Analysis
    int phase = 11;
    int error = 0;
    Real dummy = 0.0;
    
    PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase,
            &n_, a_.data(), ia_.data(), ja_.data(),
            nullptr, &nrhs_, iparm_, &error, 
            nullptr, nullptr);
    
    checkError(error, "Analysis");
    factorized_ = false;
    
    if (printLevel_ > 0) {
        LOG_INFO( "[PARDISO] Analysis complete. "
                  << "Estimated nonzeros in L+U: " << iparm_[17] << std::endl);
    }
}

inline void PardisoSolver::factorize(const SparseMatrix& A) {
    if (!initialized_) {
        analyzePattern(A);
    }
    
    // Update values if matrix structure unchanged
    const auto& eigen = A.eigen();
    std::copy(eigen.valuePtr(), eigen.valuePtr() + eigen.nonZeros(), a_.begin());
    
    // Phase 22: Numerical factorization
    int phase = 22;
    int error = 0;
    
    PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase,
            &n_, a_.data(), ia_.data(), ja_.data(),
            nullptr, &nrhs_, iparm_, &error,
            nullptr, nullptr);
    
    checkError(error, "Factorization");
    factorized_ = true;
    
    if (printLevel_ > 0) {
        LOG_INFO("[PARDISO] Factorization complete. "
                  << "MFlops: " << iparm_[18] << std::endl);
    }
}

inline bool PardisoSolver::solve(const SparseMatrix& A, Vector& x, const Vector& b) {
    if (!factorized_) {
        if (!initialized_) {
            analyzePattern(A);
        }
        factorize(A);
    }
    
    // Phase 33: Solve and iterative refinement
    int phase = 33;
    int error = 0;
    
    // Copy RHS
    std::vector<Real> rhs(b.size());
    std::copy(b.data(), b.data() + b.size(), rhs.begin());
    
    // Solve
    PARDISO(pt_, &maxfct_, &mnum_, &mtype_, &phase,
            &n_, a_.data(), ia_.data(), ja_.data(),
            nullptr, &nrhs_, iparm_, &error,
            rhs.data(), x.data());
    
    checkError(error, "Solve");
    
    iterations_ = 1;
    residual_ = iparm_[6];  // Number of iterative refinement steps
    
    if (printLevel_ > 0) {
        LOG_INFO("[PARDISO] Solve complete. "
                  << "Refinement steps: " << iparm_[6] << std::endl);
    }
    
    return error == 0;
}

inline void PardisoSolver::checkError(int error, const char* phase) {
    if (error != 0) {
        LOG_ERROR("PARDISO " << phase << " error: " << error);
        switch (error) {
            case -1: LOG_ERROR("Input inconsistent"); break;
            case -2: LOG_ERROR("Not enough memory"); break;
            case -3: LOG_ERROR("Reordering problem"); break;
            case -4: LOG_ERROR("Zero pivot, numerical factorization or iterative refinement error"); break;
            case -5: LOG_ERROR("Error in unsorted matrix"); break;
            case -6: LOG_ERROR("Preordering failed"); break;
            case -7: LOG_ERROR("Diagonal matrix problem"); break;
            case -8: LOG_ERROR("32-bit integer overflow problem"); break;
            default: LOG_ERROR("Unknown error"); break;
        }
    }
}

#endif  // MPFEM_USE_MKL

}  // namespace mpfem

#endif  // MPFEM_PARDISO_SOLVER_HPP

#ifndef MPFEM_LINEAR_SOLVER_HPP
#define MPFEM_LINEAR_SOLVER_HPP

#include "sparse_matrix.hpp"
#include "core/types.hpp"
#include <memory>
#include <string>
#include <stdexcept>

namespace mpfem {

/**
 * @brief Abstract base class for linear solvers.
 * 
 * Strategy pattern for interchangeable solver implementations.
 * Supports both direct and iterative methods.
 */
class LinearSolver {
public:
    virtual ~LinearSolver() = default;
    
    /**
     * @brief Solve the linear system Ax = b.
     * @param A Sparse matrix (must be compressed)
     * @param x Solution vector (output)
     * @param b Right-hand side vector
     * @return true if solved successfully
     */
    virtual bool solve(const SparseMatrix& A, Vector& x, const Vector& b) = 0;
    
    /**
     * @brief Analyze pattern (for direct solvers).
     * Called once before multiple solves with same sparsity pattern.
     */
    virtual void analyzePattern(const SparseMatrix& /*A*/) {}
    
    /**
     * @brief Factorize (for direct solvers).
     * Called when matrix values change but pattern stays same.
     */
    virtual void factorize(const SparseMatrix& /*A*/) {}
    
    /// Set maximum iterations (for iterative solvers)
    virtual void setMaxIterations(int iter) {
        maxIterations_ = iter;
    }
    
    /// Set relative tolerance (for iterative solvers)
    virtual void setTolerance(Real tol) {
        tolerance_ = tol;
    }
    
    /// Set absolute tolerance (for iterative solvers)
    virtual void setAbsoluteTolerance(Real tol) {
        absTolerance_ = tol;
    }
    
    /// Set print level (0 = silent, 1 = summary, 2 = verbose)
    virtual void setPrintLevel(int level) {
        printLevel_ = level;
    }
    
    /// Get number of iterations (for iterative solvers)
    virtual int iterations() const { return iterations_; }
    
    /// Get final residual (for iterative solvers)
    virtual Real residual() const { return residual_; }
    
    /// Get solver name
    virtual std::string name() const = 0;
    
protected:
    int maxIterations_ = 1000;
    Real tolerance_ = 1e-10;
    Real absTolerance_ = 1e-14;
    int printLevel_ = 0;
    int iterations_ = 0;
    Real residual_ = 0.0;
};

/**
 * @brief Solver type enumeration.
 */
enum class SolverType {
    // Direct solvers
    EigenSparseLU,       ///< Eigen SparseLU (LU factorization)
    SuperLU,        ///< SuperLU direct solver
    
    // Iterative solvers
    EigenCG,             ///< Conjugate Gradient (symmetric positive definite)
    EigenCGWithIC,       ///< CG with Incomplete Cholesky preconditioner
    EigenBiCGSTAB,       ///< BiCGSTAB (non-symmetric)
    EigenBiCGSTABWithILUT, ///< BiCGSTAB with ILUT preconditioner
    
    // Auto selection
    Auto            ///< Let solver factory choose
};

/**
 * @brief Convert solver type to string.
 */
inline std::string solverTypeToString(SolverType type) {
    switch (type) {
        case SolverType::EigenSparseLU:   return "sparse_lu";
        case SolverType::SuperLU:    return "superlu";
        case SolverType::EigenCG:         return "cg";
        case SolverType::EigenCGWithIC:   return "cg_ic";
        case SolverType::EigenBiCGSTAB:   return "bicgstab";
        case SolverType::EigenBiCGSTABWithILUT: return "bicgstab_ilut";
        default: return "unknown";
    }
}

/**
 * @brief Convert string to solver type.
 * @throws std::runtime_error if requested solver is not compiled in
 */
inline SolverType stringToSolverType(const std::string& str) {
    if (str == "eigen_sparse_lu") return SolverType::EigenSparseLU;
    if (str == "eigen_cg") return SolverType::EigenCG;
    if (str == "eigen_cg_ic") return SolverType::EigenCGWithIC;
    if (str == "eigen_bicgstab") return SolverType::EigenBiCGSTAB;
    if (str == "eigen_bicgstab_ilut") return SolverType::EigenBiCGSTABWithILUT;
    if (str == "superlu") {
#ifndef MPFEM_USE_SUPERLU
        throw std::runtime_error("SuperLU solver requested but not available. "
                                  "Rebuild with -DMPFEM_USE_SUPERLU=ON");
#endif
        return SolverType::SuperLU;
    }
    if (str == "eigen_cg") return SolverType::EigenCG;
    if (str == "eigen_cg_ic") return SolverType::EigenCGWithIC;
    if (str == "eigen_bicgstab") return SolverType::EigenBiCGSTAB;
    if (str == "eigen_bicgstab_ilut") return SolverType::EigenBiCGSTABWithILUT;
    return SolverType::Auto;  // Default to auto selection 
}

}  // namespace mpfem

#endif  // MPFEM_LINEAR_SOLVER_HPP

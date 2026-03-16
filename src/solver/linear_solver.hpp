#ifndef MPFEM_LINEAR_SOLVER_HPP
#define MPFEM_LINEAR_SOLVER_HPP

#include "sparse_matrix.hpp"
#include "core/types.hpp"
#include <memory>
#include <string>

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

}  // namespace mpfem

#endif  // MPFEM_LINEAR_SOLVER_HPP

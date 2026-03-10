/**
 * @file direct_solver.hpp
 * @brief Direct solver using Eigen's sparse LU decomposition
 */

#ifndef MPFEM_LINALG_DIRECT_SOLVER_HPP
#define MPFEM_LINALG_DIRECT_SOLVER_HPP

#include "solver_base.hpp"
#include <Eigen/SparseLU>

namespace mpfem {

/**
 * @brief Direct solver using sparse LU factorization
 * 
 * Uses Eigen's SparseLU solver which is based on SuperLU.
 * Good for general sparse matrices (not necessarily symmetric).
 */
class DirectSolver : public SolverBase {
public:
    DirectSolver() = default;
    ~DirectSolver() override = default;
    
    /**
     * @brief Analyze sparsity pattern (optional, improves performance for multiple solves)
     */
    void analyze_pattern(const SparseMatrix& A);
    
    /**
     * @brief Factorize the matrix (optional, improves performance for multiple solves)
     */
    void factorize(const SparseMatrix& A);
    
    /**
     * @brief Solve the linear system Ax = b
     * 
     * If analyze_pattern and factorize were not called, this will
     * perform both steps automatically.
     */
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;
    
    /**
     * @brief Clear stored factorization
     */
    void clear();
    
    /**
     * @brief Get memory usage estimate
     */
    size_t memory_usage() const;

private:
    Eigen::SparseLU<SparseMatrix> solver_;
    bool analyzed_ = false;
    bool factorized_ = false;
    Index n_dofs_ = 0;
};

} // namespace mpfem

#endif // MPFEM_LINALG_DIRECT_SOLVER_HPP

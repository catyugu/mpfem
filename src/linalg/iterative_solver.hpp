/**
 * @file iterative_solver.hpp
 * @brief Iterative solvers (CG, BiCGSTAB, etc.)
 */

#ifndef MPFEM_LINALG_ITERATIVE_SOLVER_HPP
#define MPFEM_LINALG_ITERATIVE_SOLVER_HPP

#include "solver_base.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace mpfem {

/**
 * @brief Preconditioner types for iterative solvers
 */
enum class PreconditionerType {
    None,           ///< No preconditioner
    Diagonal,       ///< Diagonal (Jacobi) preconditioner
    IncompleteLU,   ///< Incomplete LU factorization
    IncompleteCholesky  ///< Incomplete Cholesky (for SPD matrices)
};

/**
 * @brief Conjugate Gradient solver
 * 
 * Suitable for symmetric positive definite (SPD) matrices.
 * Uses diagonal preconditioner by default.
 */
class CGSolver : public SolverBase {
public:
    CGSolver();
    ~CGSolver() override = default;
    
    void set_preconditioner(PreconditionerType type) { precond_type_ = type; }
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;

private:
    PreconditionerType precond_type_ = PreconditionerType::Diagonal;
};

/**
 * @brief Conjugate Gradient with Geometric Multigrid preconditioner
 * 
 * Special solver for finite element problems with good convergence
 * for ill-conditioned matrices.
 */
class CGGSolver : public SolverBase {
public:
    CGGSolver();
    ~CGGSolver() override = default;
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;
    
    /**
     * @brief Set smoother type for multigrid
     */
    void set_smoother_iterations(int iter) { smoother_iter_ = iter; }

private:
    int smoother_iter_ = 3;
};

/**
 * @brief BiCGSTAB solver
 * 
 * Suitable for general non-symmetric matrices.
 * More robust than CG for non-SPD systems.
 */
class BiCGSTABSolver : public SolverBase {
public:
    BiCGSTABSolver();
    ~BiCGSTABSolver() override = default;
    
    void set_preconditioner(PreconditionerType type) { precond_type_ = type; }
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;

private:
    PreconditionerType precond_type_ = PreconditionerType::Diagonal;
};

/**
 * @brief Minimal Residual Method (MINRES)
 * 
 * Suitable for symmetric indefinite matrices.
 */
class MINRESSolver : public SolverBase {
public:
    MINRESSolver();
    ~MINRESSolver() override = default;
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;
};

/**
 * @brief Generalized Minimum Residual (GMRES)
 * 
 * Suitable for general non-symmetric matrices.
 * More memory usage but robust.
 */
class GMRESSolver : public SolverBase {
public:
    GMRESSolver();
    ~GMRESSolver() override = default;
    
    /**
     * @brief Set restart parameter (number of vectors to keep)
     */
    void set_restart(int restart) { restart_ = restart; }
    
    SolverStatus solve(const SparseMatrix& A,
                       const DynamicVector& b,
                       DynamicVector& x) override;

private:
    int restart_ = 30;
};

} // namespace mpfem

#endif // MPFEM_LINALG_ITERATIVE_SOLVER_HPP

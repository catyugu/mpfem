/**
 * @file solver_base.hpp
 * @brief Base class for linear solvers
 */

#ifndef MPFEM_LINALG_SOLVER_BASE_HPP
#define MPFEM_LINALG_SOLVER_BASE_HPP

#include "core/types.hpp"
#include <string>
#include <memory>

namespace mpfem {

/**
 * @brief Solver status after solve
 */
enum class SolverStatus {
    Success,
    Divergence,
    MaxIterationsReached,
    NumericalIssue,
    InvalidInput
};

/**
 * @brief Abstract base class for linear solvers
 * 
 * Provides a common interface for direct and iterative solvers.
 * All solvers solve the linear system Ax = b.
 */
class SolverBase {
public:
    SolverBase() = default;
    virtual ~SolverBase() = default;
    
    /**
     * @brief Solve the linear system Ax = b
     * @param A System matrix (sparse)
     * @param b Right-hand side vector
     * @param x Solution vector (output)
     * @return Solver status
     */
    virtual SolverStatus solve(const SparseMatrix& A,
                               const DynamicVector& b,
                               DynamicVector& x) = 0;
    
    /**
     * @brief Set maximum number of iterations (for iterative solvers)
     */
    virtual void set_max_iterations(int iter) { max_iterations_ = iter; }
    
    /**
     * @brief Set convergence tolerance
     */
    virtual void set_tolerance(Scalar tol) { tolerance_ = tol; }
    
    /**
     * @brief Set output level (0=silent, 1=summary, 2=verbose)
     */
    void set_print_level(int level) { print_level_ = level; }
    
    /**
     * @brief Get number of iterations performed (for iterative solvers)
     */
    virtual int iterations() const { return iterations_; }
    
    /**
     * @brief Get final residual norm
     */
    virtual Scalar residual() const { return residual_; }
    
    /**
     * @brief Get solver status from last solve
     */
    SolverStatus status() const { return status_; }
    
    /**
     * @brief Get status as string
     */
    static std::string status_string(SolverStatus s) {
        switch (s) {
            case SolverStatus::Success: return "Success";
            case SolverStatus::Divergence: return "Divergence";
            case SolverStatus::MaxIterationsReached: return "MaxIterationsReached";
            case SolverStatus::NumericalIssue: return "NumericalIssue";
            case SolverStatus::InvalidInput: return "InvalidInput";
            default: return "Unknown";
        }
    }

protected:
    int max_iterations_ = 1000;
    Scalar tolerance_ = 1e-10;
    int print_level_ = 0;
    int iterations_ = 0;
    Scalar residual_ = 0.0;
    SolverStatus status_ = SolverStatus::Success;
};

/**
 * @brief Factory function to create a solver by type name
 * @param type Solver type: "direct", "cg", "cg_gs", "bicgstab"
 * @return Unique pointer to solver
 */
std::unique_ptr<SolverBase> create_solver(const std::string& type);

} // namespace mpfem

#endif // MPFEM_LINALG_SOLVER_BASE_HPP

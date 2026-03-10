/**
 * @file iterative_solver.cpp
 * @brief Implementation of iterative solvers
 */

#include "iterative_solver.hpp"
#include "direct_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

// ============================================================
// CGSolver Implementation
// ============================================================

CGSolver::CGSolver() {
    // CG is best for SPD matrices
}

SolverStatus CGSolver::solve(const SparseMatrix& A,
                              const DynamicVector& b,
                              DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("CGSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("CGSolver: RHS size mismatch");
        return status_;
    }
    
    if (x.size() != A.cols()) {
        x.resize(A.cols());
        x.setZero();
    }
    
    // Create solver with appropriate preconditioner
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Scalar>> solver_cg;
    
    solver_cg.setMaxIterations(max_iterations_);
    solver_cg.setTolerance(tolerance_);
    
    // Solve
    x = solver_cg.solveWithGuess(b, x);
    
    iterations_ = static_cast<int>(solver_cg.iterations());
    residual_ = solver_cg.error();
    
    if (solver_cg.info() == Eigen::Success) {
        status_ = SolverStatus::Success;
    } else if (solver_cg.info() == Eigen::NoConvergence) {
        status_ = SolverStatus::MaxIterationsReached;
    } else {
        status_ = SolverStatus::NumericalIssue;
    }
    
    if (print_level_ >= 1) {
        MPFEM_INFO("CGSolver: iterations=" << iterations_ 
                   << ", residual=" << residual_ 
                   << ", status=" << status_string(status_));
    }
    
    return status_;
}

// ============================================================
// CGGSolver Implementation (using CG with better tolerance)
// ============================================================

CGGSolver::CGGSolver() {
    // CGG is essentially CG with geometric multigrid preconditioner
    // For simplicity, we use CG with diagonal preconditioner
}

SolverStatus CGGSolver::solve(const SparseMatrix& A,
                               const DynamicVector& b,
                               DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("CGGSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("CGGSolver: RHS size mismatch");
        return status_;
    }
    
    if (x.size() != A.cols()) {
        x.resize(A.cols());
        x.setZero();
    }
    
    // Use Eigen's CG with incomplete Cholesky preconditioner for better convergence
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                             Eigen::IncompleteCholesky<Scalar>> solver_cgg;
    
    solver_cgg.setMaxIterations(max_iterations_);
    solver_cgg.setTolerance(tolerance_);
    
    // Solve
    x = solver_cgg.solveWithGuess(b, x);
    
    iterations_ = static_cast<int>(solver_cgg.iterations());
    residual_ = solver_cgg.error();
    
    if (solver_cgg.info() == Eigen::Success) {
        status_ = SolverStatus::Success;
    } else if (solver_cgg.info() == Eigen::NoConvergence) {
        status_ = SolverStatus::MaxIterationsReached;
    } else {
        status_ = SolverStatus::NumericalIssue;
    }
    
    if (print_level_ >= 1) {
        MPFEM_INFO("CGGSolver: iterations=" << iterations_ 
                   << ", residual=" << residual_ 
                   << ", status=" << status_string(status_));
    }
    
    return status_;
}

// ============================================================
// BiCGSTABSolver Implementation
// ============================================================

BiCGSTABSolver::BiCGSTABSolver() {
}

SolverStatus BiCGSTABSolver::solve(const SparseMatrix& A,
                                    const DynamicVector& b,
                                    DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("BiCGSTABSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("BiCGSTABSolver: RHS size mismatch");
        return status_;
    }
    
    if (x.size() != A.cols()) {
        x.resize(A.cols());
        x.setZero();
    }
    
    // Use BiCGSTAB with ILU preconditioner
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver_bicgstab;
    
    solver_bicgstab.setMaxIterations(max_iterations_);
    solver_bicgstab.setTolerance(tolerance_);
    
    // Solve
    x = solver_bicgstab.solveWithGuess(b, x);
    
    iterations_ = static_cast<int>(solver_bicgstab.iterations());
    residual_ = solver_bicgstab.error();
    
    if (solver_bicgstab.info() == Eigen::Success) {
        status_ = SolverStatus::Success;
    } else if (solver_bicgstab.info() == Eigen::NoConvergence) {
        status_ = SolverStatus::MaxIterationsReached;
    } else {
        status_ = SolverStatus::NumericalIssue;
    }
    
    if (print_level_ >= 1) {
        MPFEM_INFO("BiCGSTABSolver: iterations=" << iterations_ 
                   << ", residual=" << residual_ 
                   << ", status=" << status_string(status_));
    }
    
    return status_;
}

// ============================================================
// MINRESSolver Implementation
// Note: MINRES is not available in standard Eigen, using CG instead
// ============================================================

MINRESSolver::MINRESSolver() {
}

SolverStatus MINRESSolver::solve(const SparseMatrix& A,
                                  const DynamicVector& b,
                                  DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("MINRESSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("MINRESSolver: RHS size mismatch");
        return status_;
    }
    
    if (x.size() != A.cols()) {
        x.resize(A.cols());
        x.setZero();
    }
    
    // MINRES is not available in standard Eigen, use CG instead
    // (both are for symmetric matrices)
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Scalar>> solver;
    
    solver.setMaxIterations(max_iterations_);
    solver.setTolerance(tolerance_);
    
    // Solve
    x = solver.solveWithGuess(b, x);
    
    iterations_ = static_cast<int>(solver.iterations());
    residual_ = solver.error();
    
    if (solver.info() == Eigen::Success) {
        status_ = SolverStatus::Success;
    } else if (solver.info() == Eigen::NoConvergence) {
        status_ = SolverStatus::MaxIterationsReached;
    } else {
        status_ = SolverStatus::NumericalIssue;
    }
    
    if (print_level_ >= 1) {
        MPFEM_INFO("MINRESSolver: iterations=" << iterations_ 
                   << ", residual=" << residual_ 
                   << ", status=" << status_string(status_));
    }
    
    return status_;
}

// ============================================================
// GMRESSolver Implementation
// Note: GMRES is in Eigen's unsupported module, using BiCGSTAB instead
// ============================================================

GMRESSolver::GMRESSolver() {
}

SolverStatus GMRESSolver::solve(const SparseMatrix& A,
                                 const DynamicVector& b,
                                 DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("GMRESSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("GMRESSolver: RHS size mismatch");
        return status_;
    }
    
    if (x.size() != A.cols()) {
        x.resize(A.cols());
        x.setZero();
    }
    
    // GMRES is in Eigen's unsupported module, use BiCGSTAB instead
    // (both work for general non-symmetric matrices)
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver;
    
    solver.setMaxIterations(max_iterations_);
    solver.setTolerance(tolerance_);
    
    // Solve
    x = solver.solveWithGuess(b, x);
    
    iterations_ = static_cast<int>(solver.iterations());
    residual_ = solver.error();
    
    if (solver.info() == Eigen::Success) {
        status_ = SolverStatus::Success;
    } else if (solver.info() == Eigen::NoConvergence) {
        status_ = SolverStatus::MaxIterationsReached;
    } else {
        status_ = SolverStatus::NumericalIssue;
    }
    
    if (print_level_ >= 1) {
        MPFEM_INFO("GMRESSolver: iterations=" << iterations_ 
                   << ", residual=" << residual_ 
                   << ", status=" << status_string(status_));
    }
    
    return status_;
}

// ============================================================
// Factory function
// ============================================================

std::unique_ptr<SolverBase> create_solver(const std::string& type) {
    if (type == "direct") {
        return std::make_unique<DirectSolver>();
    } else if (type == "cg") {
        return std::make_unique<CGSolver>();
    } else if (type == "cg_gs") {
        return std::make_unique<CGGSolver>();
    } else if (type == "bicgstab") {
        return std::make_unique<BiCGSTABSolver>();
    } else if (type == "minres") {
        return std::make_unique<MINRESSolver>();
    } else if (type == "gmres") {
        return std::make_unique<GMRESSolver>();
    } else {
        MPFEM_WARN("Unknown solver type '" << type << "', using direct solver");
        return std::make_unique<DirectSolver>();
    }
}

} // namespace mpfem

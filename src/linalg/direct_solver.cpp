/**
 * @file direct_solver.cpp
 * @brief Implementation of direct solver
 */

#include "direct_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

void DirectSolver::analyze_pattern(const SparseMatrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("DirectSolver: Empty matrix");
        return;
    }
    
    n_dofs_ = A.rows();
    solver_.analyzePattern(A);
    
    if (solver_.info() != Eigen::Success) {
        status_ = SolverStatus::NumericalIssue;
        MPFEM_ERROR("DirectSolver: Pattern analysis failed");
        return;
    }
    
    analyzed_ = true;
    
    if (print_level_ >= 1) {
        MPFEM_INFO("DirectSolver: Pattern analysis complete, size=" << n_dofs_);
    }
}

void DirectSolver::factorize(const SparseMatrix& A) {
    if (!analyzed_) {
        analyze_pattern(A);
        if (status_ != SolverStatus::Success) {
            return;
        }
    }
    
    solver_.factorize(A);
    
    if (solver_.info() != Eigen::Success) {
        status_ = SolverStatus::NumericalIssue;
        MPFEM_ERROR("DirectSolver: Factorization failed");
        return;
    }
    
    factorized_ = true;
    
    if (print_level_ >= 1) {
        MPFEM_INFO("DirectSolver: Factorization complete");
    }
}

SolverStatus DirectSolver::solve(const SparseMatrix& A,
                                  const DynamicVector& b,
                                  DynamicVector& x) {
    // Validate input
    if (A.rows() == 0 || A.cols() == 0) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("DirectSolver: Empty matrix");
        return status_;
    }
    
    if (b.size() != A.rows()) {
        status_ = SolverStatus::InvalidInput;
        MPFEM_ERROR("DirectSolver: RHS size mismatch: b.size()=" << b.size() << " A.rows()=" << A.rows());
        return status_;
    }
    
    MPFEM_INFO("DirectSolver: Matrix size " << A.rows() << "x" << A.cols() 
               << ", non-zeros=" << A.nonZeros());
    
    // Resize solution vector if needed
    if (x.size() != A.cols()) {
        x.resize(A.cols());
    }
    
    // Perform factorization if not done
    if (!factorized_) {
        if (!analyzed_) {
            analyze_pattern(A);
        }
        factorize(A);
        
        if (status_ != SolverStatus::Success) {
            MPFEM_ERROR("DirectSolver: Factorization failed with status " << static_cast<int>(status_));
            return status_;
        }
    }
    
    // Solve
    x = solver_.solve(b);
    
    if (solver_.info() != Eigen::Success) {
        status_ = SolverStatus::NumericalIssue;
        MPFEM_ERROR("DirectSolver: Solve failed");
        return status_;
    }
    
    // Compute residual
    DynamicVector r = b - A * x;
    residual_ = r.norm();
    
    status_ = SolverStatus::Success;
    iterations_ = 1;  // Direct solver uses "1 iteration"
    
    if (print_level_ >= 1) {
        MPFEM_INFO("DirectSolver: Solve complete, residual=" << residual_);
    }
    
    return status_;
}

void DirectSolver::clear() {
    solver_.~SparseLU();
    new (&solver_) Eigen::SparseLU<SparseMatrix>();
    analyzed_ = false;
    factorized_ = false;
    n_dofs_ = 0;
}

size_t DirectSolver::memory_usage() const {
    // Approximate memory usage (Eigen doesn't provide exact value)
    return static_cast<size_t>(n_dofs_) * sizeof(Scalar) * 10;  // Rough estimate
}

} // namespace mpfem

#include "mpfem/solver/linear_solver.hpp"
#include "mpfem/core/logger.hpp"

namespace mpfem {

// ============================================================================
// SparseLUSolver implementation
// ============================================================================

void SparseLUSolver::SetOperator(const SparseMatrix& A) {
  MPFEM_INFO("SparseLUSolver: computing factorization for %dx%d matrix with %d non-zeros",
             static_cast<int>(A.rows()), static_cast<int>(A.cols()),
             static_cast<int>(A.nonZeros()));

  // Check for empty matrix
  if (A.rows() == 0 || A.cols() == 0 || A.nonZeros() == 0) {
    MPFEM_ERROR("SparseLU: matrix is empty");
    info_ = -1;
    return;
  }

  // Make a copy and ensure it's compressed
  SparseMatrix A_copy = A;
  A_copy.makeCompressed();

  // Use compute() which handles analyzePattern + factorize internally
  solver_.compute(A_copy);
  info_ = static_cast<int>(solver_.info());

  if (info_ != Eigen::Success) {
    MPFEM_ERROR("SparseLU compute failed with info = %d", info_);
  } else {
    MPFEM_INFO("SparseLU factorization successful");
  }
}

void SparseLUSolver::Mult(const VectorXd& b, VectorXd& x) const {
  x = solver_.solve(b);
  info_ = static_cast<int>(solver_.info());

  if (info_ != Eigen::Success) {
    MPFEM_ERROR("SparseLU solve failed with info = %d", info_);
  }
}

// ============================================================================
// SparseCGSolver implementation
// ============================================================================

void SparseCGSolver::SetOperator(const SparseMatrix& A) {
  MPFEM_INFO("CGSolver: setting up matrix %dx%d with %d non-zeros",
             static_cast<int>(A.rows()), static_cast<int>(A.cols()),
             static_cast<int>(A.nonZeros()));

  SparseMatrix A_copy = A;
  A_copy.makeCompressed();

  solver_.compute(A_copy);
  solver_.setMaxIterations(max_iter_);
  solver_.setTolerance(rel_tol_);

  if (solver_.info() != Eigen::Success) {
    MPFEM_ERROR("CG preconditioner setup failed with info = %d",
                static_cast<int>(solver_.info()));
  } else {
    MPFEM_INFO("CG preconditioner setup successful");
  }
}

void SparseCGSolver::Mult(const VectorXd& b, VectorXd& x) const {
  x = solver_.solveWithGuess(b, x);
  iterations_ = solver_.iterations();
  error_ = solver_.error();

  if (solver_.info() != Eigen::Success) {
    MPFEM_WARN("CG solve did not converge: iterations = %d, error = %g",
               iterations_, error_);
  } else {
    MPFEM_INFO("CG solve converged: iterations = %d, error = %g",
               iterations_, error_);
  }
}

// ============================================================================
// BiCGSTABSolver implementation
// ============================================================================

void BiCGSTABSolver::SetOperator(const SparseMatrix& A) {
  MPFEM_INFO("BiCGSTABSolver: setting up matrix %dx%d with %d non-zeros",
             static_cast<int>(A.rows()), static_cast<int>(A.cols()),
             static_cast<int>(A.nonZeros()));

  SparseMatrix A_copy = A;
  A_copy.makeCompressed();

  solver_.compute(A_copy);
  solver_.setMaxIterations(max_iter_);
  solver_.setTolerance(rel_tol_);

  if (solver_.info() != Eigen::Success) {
    MPFEM_ERROR("BiCGSTAB preconditioner setup failed with info = %d",
                static_cast<int>(solver_.info()));
  } else {
    MPFEM_INFO("BiCGSTAB preconditioner setup successful");
  }
}

void BiCGSTABSolver::Mult(const VectorXd& b, VectorXd& x) const {
  x = solver_.solveWithGuess(b, x);
  iterations_ = solver_.iterations();
  error_ = solver_.error();

  if (solver_.info() != Eigen::Success) {
    MPFEM_WARN("BiCGSTAB solve did not converge: iterations = %d, error = %g",
               iterations_, error_);
  } else {
    MPFEM_INFO("BiCGSTAB solve converged: iterations = %d, error = %g",
               iterations_, error_);
  }
}

}  // namespace mpfem
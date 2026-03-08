#pragma once

#include <memory>

#include "mpfem/core/eigen_types.hpp"

namespace mpfem {

// ============================================================================
// LinearSolver - 线性求解器基类
// ============================================================================

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;

  // Set the matrix
  virtual void SetOperator(const SparseMatrix& A) = 0;

  // Solve the system Ax = b
  virtual void Mult(const VectorXd& b, VectorXd& x) const = 0;

  // Solve in-place
  void Solve(const SparseMatrix& A, const VectorXd& b, VectorXd& x) {
    SetOperator(A);
    Mult(b, x);
  }
};

// ============================================================================
// SparseLUSolver - Eigen SparseLU 直接求解器
// ============================================================================

class SparseLUSolver : public LinearSolver {
 public:
  SparseLUSolver() = default;

  void SetOperator(const SparseMatrix& A) override;

  void Mult(const VectorXd& b, VectorXd& x) const override;

  // Get solver info
  int GetInfo() const { return info_; }

 private:
  Eigen::SparseLU<SparseMatrix> solver_;
  mutable int info_ = 0;
};

// ============================================================================
// SparseCGSolver - Eigen Conjugate Gradient 迭代求解器
// ============================================================================

class SparseCGSolver : public LinearSolver {
 public:
  SparseCGSolver() : max_iter_(1000), rel_tol_(1e-10), abs_tol_(1e-15) {}

  void SetOperator(const SparseMatrix& A) override;
  void Mult(const VectorXd& b, VectorXd& x) const override;

  void SetMaxIterations(int max_iter) { max_iter_ = max_iter; }
  void SetRelativeTolerance(double rel_tol) { rel_tol_ = rel_tol; }
  void SetAbsoluteTolerance(double abs_tol) { abs_tol_ = abs_tol; }

  int GetIterations() const { return iterations_; }
  double GetError() const { return error_; }

 private:
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver_;
  int max_iter_;
  double rel_tol_;
  double abs_tol_;
  mutable int iterations_ = 0;
  mutable double error_ = 0.0;
};

// ============================================================================
// BiCGSTABSolver - Eigen BiCGSTAB 迭代求解器（用于非对称矩阵）
// ============================================================================

class BiCGSTABSolver : public LinearSolver {
 public:
  BiCGSTABSolver() : max_iter_(1000), rel_tol_(1e-10) {}

  void SetOperator(const SparseMatrix& A) override;
  void Mult(const VectorXd& b, VectorXd& x) const override;

  void SetMaxIterations(int max_iter) { max_iter_ = max_iter; }
  void SetRelativeTolerance(double rel_tol) { rel_tol_ = rel_tol; }

  int GetIterations() const { return iterations_; }
  double GetError() const { return error_; }

 private:
  Eigen::BiCGSTAB<SparseMatrix> solver_;
  int max_iter_;
  double rel_tol_;
  mutable int iterations_ = 0;
  mutable double error_ = 0.0;
};

// ============================================================================
// SolverFactory - 求解器工厂
// ============================================================================

class SolverFactory {
 public:
  enum class SolverType {
    kSparseLU,
    kCG,
    kBiCGSTAB
  };

  static std::unique_ptr<LinearSolver> Create(SolverType type) {
    switch (type) {
      case SolverType::kSparseLU:
        return std::make_unique<SparseLUSolver>();
      case SolverType::kCG:
        return std::make_unique<SparseCGSolver>();
      case SolverType::kBiCGSTAB:
        return std::make_unique<BiCGSTABSolver>();
      default:
        return std::make_unique<SparseLUSolver>();
    }
  }
};

}  // namespace mpfem
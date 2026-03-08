#pragma once

#include <memory>
#include <vector>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/fem/fe_space.hpp"
#include "mpfem/fem/integrator.hpp"

namespace mpfem {

// ============================================================================
// BilinearForm - 双线性形式
// 组装稀疏刚度矩阵
// ============================================================================

class BilinearForm {
 public:
  explicit BilinearForm(FiniteElementSpace* fes);
  ~BilinearForm() = default;

  // Add domain integrator
  void AddDomainIntegrator(std::unique_ptr<BilinearFormIntegrator> integrator,
                           const std::vector<int>& domain_ids = {});

  // Add boundary integrator (for Robin boundary conditions)
  void AddBoundaryIntegrator(std::unique_ptr<BilinearFormIntegrator> integrator,
                             const std::vector<int>& boundary_ids);

  // Assemble the global matrix
  void Assemble();

  // Form the linear system, eliminating essential boundary conditions
  // A: system matrix (with essential DOFs eliminated)
  // X: solution vector for true DOFs
  // B: RHS vector for true DOFs
  // x: initial guess (contains Dirichlet BC values)
  // b: RHS vector
  void FormLinearSystem(const std::vector<int>& ess_dofs,
                        VectorXd& x, VectorXd& b,
                        SparseMatrix& A, VectorXd& X, VectorXd& B);

  // Get the assembled matrix
  const SparseMatrix& GetMatrix() const { return mat_; }
  SparseMatrix& GetMatrix() { return mat_; }

  // Get the full matrix (before boundary condition elimination)
  const SparseMatrix& GetFullMatrix() const { return mat_full_; }

  // Get the FE space
  FiniteElementSpace* GetFES() const { return fes_; }

 private:
  FiniteElementSpace* fes_;
  SparseMatrix mat_;         // Assembled matrix (after BC elimination)
  SparseMatrix mat_full_;    // Full matrix (before BC elimination)

  std::vector<DomainIntegrator> domain_integrators_;
  std::vector<BoundaryIntegrator> boundary_integrators_;
};

// ============================================================================
// LinearForm - 线性形式
// 组装右端向量
// ============================================================================

class LinearForm {
 public:
  explicit LinearForm(FiniteElementSpace* fes);
  ~LinearForm() = default;

  // Add domain integrator (for source terms)
  void AddDomainIntegrator(std::unique_ptr<LinearFormIntegrator> integrator,
                           const std::vector<int>& domain_ids = {});

  // Add boundary integrator (for Neumann BC, convection RHS)
  void AddBoundaryIntegrator(std::unique_ptr<LinearFormIntegrator> integrator,
                             const std::vector<int>& boundary_ids);

  // Assemble the global vector
  void Assemble();

  // Get the assembled vector
  const VectorXd& GetVector() const { return vec_; }
  VectorXd& GetVector() { return vec_; }

  // Element access
  double& operator()(int i) { return vec_(i); }
  double operator()(int i) const { return vec_(i); }

  // Get the FE space
  FiniteElementSpace* GetFES() const { return fes_; }

 private:
  FiniteElementSpace* fes_;
  VectorXd vec_;

  struct DomainLFIntegrator {
    std::unique_ptr<LinearFormIntegrator> integrator;
    std::vector<int> domain_ids;
  };

  struct BoundaryLFIntegrator {
    std::unique_ptr<LinearFormIntegrator> integrator;
    std::vector<int> boundary_ids;
  };

  std::vector<DomainLFIntegrator> domain_integrators_;
  std::vector<BoundaryLFIntegrator> boundary_integrators_;
};

}  // namespace mpfem
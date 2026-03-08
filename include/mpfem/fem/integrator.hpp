#pragma once

#include <memory>

#include "mpfem/core/eigen_types.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/fem/fe_space.hpp"
#include "mpfem/fem/coefficient.hpp"

namespace mpfem {

// Forward declarations
class ElementTransformation;

// ============================================================================
// BilinearFormIntegrator - 双线性形式积分器基类
// ============================================================================

class BilinearFormIntegrator {
 public:
  virtual ~BilinearFormIntegrator() = default;

  // Assemble the element matrix
  // elmat is a dof x dof matrix
  virtual void AssembleElementMatrix(const FiniteElement& el,
                                      ElementTransformation& T,
                                      Eigen::MatrixXd& elmat) = 0;

  // Assemble the boundary face matrix (for boundary conditions)
  virtual void AssembleFaceMatrix(const FiniteElement& el,
                                   ElementTransformation& T,
                                   Eigen::MatrixXd& elmat) {
    (void)el;
    (void)T;
    elmat.setZero();
  }

  // Get the required integration order
  virtual int GetIntegrationOrder() const { return 0; }
};

// ============================================================================
// LinearFormIntegrator - 线性形式积分器基类
// ============================================================================

class LinearFormIntegrator {
 public:
  virtual ~LinearFormIntegrator() = default;

  // Assemble the element vector
  // elvec is a dof x 1 vector
  virtual void AssembleElementVector(const FiniteElement& el,
                                      ElementTransformation& T,
                                      Eigen::VectorXd& elvec) = 0;

  // Assemble the boundary element vector
  virtual void AssembleBoundaryVector(const FiniteElement& el,
                                       ElementTransformation& T,
                                       Eigen::VectorXd& elvec) {
    (void)el;
    (void)T;
    elvec.setZero();
  }

  // Get the required integration order
  virtual int GetIntegrationOrder() const { return 0; }
};

// ============================================================================
// DomainIntegrator - 区域积分器辅助类
// ============================================================================

struct DomainIntegrator {
  std::unique_ptr<BilinearFormIntegrator> integrator;
  std::vector<int> domain_ids;  // Empty means all domains

  bool AppliesToDomain(int domain_id) const {
    if (domain_ids.empty()) return true;
    return std::find(domain_ids.begin(), domain_ids.end(), domain_id) !=
           domain_ids.end();
  }
};

// ============================================================================
// BoundaryIntegrator - 边界积分器辅助类
// ============================================================================

struct BoundaryIntegrator {
  std::unique_ptr<BilinearFormIntegrator> integrator;
  std::unique_ptr<LinearFormIntegrator> linear_integrator;
  std::vector<int> boundary_ids;

  bool AppliesToBoundary(int boundary_id) const {
    if (boundary_ids.empty()) return true;
    return std::find(boundary_ids.begin(), boundary_ids.end(), boundary_id) !=
           boundary_ids.end();
  }
};

}  // namespace mpfem

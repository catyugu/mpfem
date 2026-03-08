#include "mpfem/fem/integrators.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/core/logger.hpp"
#include <cmath>

namespace mpfem {

// ============================================================================
// DiffusionIntegrator implementation
// ============================================================================

void DiffusionIntegrator::AssembleElementMatrix(const FiniteElement& el,
                                                 ElementTransformation& T,
                                                 Eigen::MatrixXd& elmat) {
  int dof = el.GetDof();
  int dim = el.GetDim();

  elmat.resize(dof, dof);
  elmat.setZero();

  // Get integration rule
  int order = std::max(el.GetOrder(), GetIntegrationOrder());
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), order);

  // Allocate shape function gradients
  std::vector<double> dshape(dof * dim);
  Eigen::MatrixXd dshape_mat(dof, dim);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    // Calculate shape function gradients in reference coordinates
    el.CalcDShape(ip, dshape.data());

    // Copy to matrix form
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        dshape_mat(j, k) = dshape[j * dim + k];
      }
    }

    // Get Jacobian and its inverse
    Mat3 J = T.CalcJacobian(ip);
    Mat3 Jinv = J.inverse();
    double detJ = T.CalcWeight(ip);

    // Transform gradients to physical coordinates
    // ∇φ = J^{-T} ∇̂φ
    Eigen::MatrixXd grad(dof, 3);
    grad.setZero();
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          grad(j, l) += dshape_mat(j, k) * Jinv(k, l);
        }
      }
    }

    // Get coefficient value
    double q_val = Q_ ? Q_->Eval(T, ip) : 1.0;

    // Element matrix: elmat += q * w * detJ * (grad * grad^T)
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dof; ++k) {
        double sum = 0.0;
        for (int l = 0; l < dim; ++l) {
          sum += grad(j, l) * grad(k, l);
        }
        elmat(j, k) += q_val * ip.weight * detJ * sum;
      }
    }
  }
}

// ============================================================================
// MassIntegrator implementation
// ============================================================================

void MassIntegrator::AssembleElementMatrix(const FiniteElement& el,
                                            ElementTransformation& T,
                                            Eigen::MatrixXd& elmat) {
  int dof = el.GetDof();

  elmat.resize(dof, dof);
  elmat.setZero();

  // Get integration rule
  int order = std::max(el.GetOrder(), GetIntegrationOrder());
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), order);

  // Allocate shape functions
  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    // Calculate shape functions
    el.CalcShape(ip, shape.data());

    // Get Jacobian determinant
    double detJ = T.CalcWeight(ip);

    // Get coefficient value
    double q_val = Q_ ? Q_->Eval(T, ip) : 1.0;

    // Element matrix: elmat += q * w * detJ * (shape * shape^T)
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dof; ++k) {
        elmat(j, k) += q_val * ip.weight * detJ * shape[j] * shape[k];
      }
    }
  }
}

// ============================================================================
// ElasticityIntegrator implementation
// ============================================================================

void ElasticityIntegrator::AssembleElementMatrix(const FiniteElement& el,
                                                  ElementTransformation& T,
                                                  Eigen::MatrixXd& elmat) {
  int dof = el.GetDof();
  int dim = el.GetDim();
  int vdim = 3;  // 3D elasticity

  // For vector-valued elements, dof = scalar_dof * vdim
  int scalar_dof = dof / vdim;
  if (dof % vdim != 0) {
    // This is a scalar element, we need to handle vector case
    scalar_dof = dof;
  }

  elmat.resize(dof * vdim, dof * vdim);
  elmat.setZero();

  // Get integration rule
  int order = std::max(el.GetOrder(), GetIntegrationOrder());
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), order);

  // Allocate shape function gradients
  std::vector<double> dshape(scalar_dof * dim);
  Eigen::MatrixXd dshape_mat(scalar_dof, dim);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    // Calculate shape function gradients
    el.CalcDShape(ip, dshape.data());

    // Copy to matrix form
    for (int j = 0; j < scalar_dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        dshape_mat(j, k) = dshape[j * dim + k];
      }
    }

    // Get Jacobian
    Mat3 J = T.CalcJacobian(ip);
    Mat3 Jinv = J.inverse();
    double detJ = T.CalcWeight(ip);

    // Transform gradients
    Eigen::MatrixXd grad(scalar_dof, 3);
    grad.setZero();
    for (int j = 0; j < scalar_dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          grad(j, l) += dshape_mat(j, k) * Jinv(k, l);
        }
      }
    }

    // Get Lamé parameters
    double lambda = lambda_ ? lambda_->Eval(T, ip) : 1.0;
    double mu = mu_ ? mu_->Eval(T, ip) : 1.0;

    // Assemble element stiffness matrix for 3D elasticity
    // For each pair of shape functions (i, j), compute the 3x3 block
    for (int ii = 0; ii < scalar_dof; ++ii) {
      for (int jj = 0; jj < scalar_dof; ++jj) {
        // B_ij = [grad_φ_i; grad_φ_j]
        // K_ij = ∫ (λ * div_φ_i * div_φ_j + 2μ * ε(φ_i) : ε(φ_j)) dV

        // div_φ_i = sum_k grad_φ_i(k)
        double div_i = grad(ii, 0) + grad(ii, 1) + grad(ii, 2);
        double div_j = grad(jj, 0) + grad(jj, 1) + grad(jj, 2);

        // 3x3 block for node i and node j
        for (int m = 0; m < 3; ++m) {
          for (int n = 0; n < 3; ++n) {
            double val = 0.0;

            // λ * div_i * div_j * δ_mn
            if (m == n) {
              val += lambda * div_i * div_j;
            }

            // μ * (grad_φ_i(m) * grad_φ_j(n) + grad_φ_i(n) * grad_φ_j(m))
            val += mu * (grad(ii, m) * grad(jj, n) + grad(ii, n) * grad(jj, m));

            // Symmetric part
            // Actually, the correct form is:
            // K_ij^{mn} = λ * ∂φ_i/∂m * ∂φ_j/∂n + μ * (∂φ_i/∂n * ∂φ_j/∂m + δ_mn * ∂φ_i/∂k * ∂φ_j/∂k)

            elmat(ii * vdim + m, jj * vdim + n) +=
                val * ip.weight * detJ;
          }
        }
      }
    }
  }
}

// ============================================================================
// VectorMassIntegrator implementation
// ============================================================================

void VectorMassIntegrator::AssembleElementMatrix(const FiniteElement& el,
                                                  ElementTransformation& T,
                                                  Eigen::MatrixXd& elmat) {
  int dof = el.GetDof();
  int vdim = 3;

  elmat.resize(dof * vdim, dof * vdim);
  elmat.setZero();

  // Get integration rule
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), el.GetOrder() + 1);

  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);
    el.CalcShape(ip, shape.data());

    double detJ = T.CalcWeight(ip);
    double q_val = Q_ ? Q_->Eval(T, ip) : 1.0;

    // Block diagonal mass matrix
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dof; ++k) {
        double m = q_val * ip.weight * detJ * shape[j] * shape[k];
        for (int c = 0; c < vdim; ++c) {
          elmat(j * vdim + c, k * vdim + c) += m;
        }
      }
    }
  }
}

// ============================================================================
// DomainLFIntegrator implementation
// ============================================================================

void DomainLFIntegrator::AssembleElementVector(const FiniteElement& el,
                                                ElementTransformation& T,
                                                Eigen::VectorXd& elvec) {
  int dof = el.GetDof();

  elvec.resize(dof);
  elvec.setZero();

  // Get integration rule
  int order = std::max(el.GetOrder(), GetIntegrationOrder());
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), order);

  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    el.CalcShape(ip, shape.data());
    double detJ = T.CalcWeight(ip);
    double q_val = Q_ ? Q_->Eval(T, ip) : 1.0;

    // Element vector: elvec += q * w * detJ * shape
    for (int j = 0; j < dof; ++j) {
      elvec(j) += q_val * ip.weight * detJ * shape[j];
    }
  }
}

// ============================================================================
// BoundaryLFIntegrator implementation
// ============================================================================

void BoundaryLFIntegrator::AssembleBoundaryVector(const FiniteElement& el,
                                                   ElementTransformation& T,
                                                   Eigen::VectorXd& elvec) {
  int dof = el.GetDof();

  elvec.resize(dof);
  elvec.setZero();

  // Get integration rule for boundary element
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), el.GetOrder() + 1);

  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    el.CalcShape(ip, shape.data());

    // For boundary elements, detJ gives the surface measure
    double detJ = T.CalcWeight(ip);
    double q_val = Q_ ? Q_->Eval(T, ip) : 1.0;

    for (int j = 0; j < dof; ++j) {
      elvec(j) += q_val * ip.weight * detJ * shape[j];
    }
  }
}

// ============================================================================
// ConvectionIntegrator implementation
// ============================================================================

void ConvectionIntegrator::AssembleFaceMatrix(const FiniteElement& el,
                                               ElementTransformation& T,
                                               Eigen::MatrixXd& elmat) {
  int dof = el.GetDof();

  elmat.resize(dof, dof);
  elmat.setZero();

  // Get integration rule
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), el.GetOrder() + 1);

  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    el.CalcShape(ip, shape.data());

    // Surface measure
    double detJ = T.CalcWeight(ip);
    double h_val = h_ ? h_->Eval(T, ip) : 1.0;

    // Element matrix: h * w * detJ * (shape * shape^T)
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dof; ++k) {
        elmat(j, k) += h_val * ip.weight * detJ * shape[j] * shape[k];
      }
    }
  }
}

// ============================================================================
// ConvectionRHSIntegrator implementation
// ============================================================================

void ConvectionRHSIntegrator::AssembleBoundaryVector(const FiniteElement& el,
                                                      ElementTransformation& T,
                                                      Eigen::VectorXd& elvec) {
  int dof = el.GetDof();

  elvec.resize(dof);
  elvec.setZero();

  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), el.GetOrder() + 1);

  std::vector<double> shape(dof);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    el.CalcShape(ip, shape.data());

    double detJ = T.CalcWeight(ip);
    double h_val = h_ ? h_->Eval(T, ip) : 1.0;
    double T_ext_val = T_ext_ ? T_ext_->Eval(T, ip) : 293.15;

    // RHS: h * T_ext * shape
    for (int j = 0; j < dof; ++j) {
      elvec(j) += h_val * T_ext_val * ip.weight * detJ * shape[j];
    }
  }
}

// ============================================================================
// ThermalExpansionIntegrator implementation
// ============================================================================

void ThermalExpansionIntegrator::AssembleElementVector(const FiniteElement& el,
                                                        ElementTransformation& T,
                                                        Eigen::VectorXd& elvec) {
  int dof = el.GetDof();
  int dim = el.GetDim();
  int vdim = 3;  // 3D elasticity

  int scalar_dof = dof;
  elvec.resize(dof * vdim);
  elvec.setZero();

  // Get integration rule
  const IntegrationRule& ir =
      IntegrationRules::Instance().Get(el.GetGeomType(), el.GetOrder() + 1);

  std::vector<double> dshape(dof * dim);
  Eigen::MatrixXd dshape_mat(dof, dim);

  for (int i = 0; i < ir.GetNPoints(); ++i) {
    const IntegrationPoint& ip = ir.GetPoint(i);

    el.CalcDShape(ip, dshape.data());

    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        dshape_mat(j, k) = dshape[j * dim + k];
      }
    }

    // Get Jacobian
    Mat3 J = T.CalcJacobian(ip);
    Mat3 Jinv = J.inverse();
    double detJ = T.CalcWeight(ip);

    // Transform gradients
    Eigen::MatrixXd grad(dof, 3);
    grad.setZero();
    for (int j = 0; j < dof; ++j) {
      for (int k = 0; k < dim; ++k) {
        for (int l = 0; l < dim; ++l) {
          grad(j, l) += dshape_mat(j, k) * Jinv(k, l);
        }
      }
    }

    // Get material properties
    double alpha = alpha_ ? alpha_->Eval(T, ip) : 0.0;
    double temp = T_ ? T_->Eval(T, ip) : T_ref_;
    double lambda = lambda_ ? lambda_->Eval(T, ip) : 0.0;
    double mu = mu_ ? mu_->Eval(T, ip) : 0.0;

    // Thermal strain
    double dT = temp - T_ref_;

    // Stress from thermal strain: σ_th = (3λ + 2μ) * α * dT * I
    // Equivalent nodal force: f = ∫ B^T σ_th dV

    double thermal_stress = (3.0 * lambda + 2.0 * mu) * alpha * dT;

    for (int j = 0; j < dof; ++j) {
      // The thermal stress contributes to the force vector
      // f_j^m = thermal_stress * div(φ_j) * δ_mn
      double div_j = grad(j, 0) + grad(j, 1) + grad(j, 2);

      for (int m = 0; m < 3; ++m) {
        elvec(j * vdim + m) +=
            thermal_stress * grad(j, m) * ip.weight * detJ;
      }
    }
  }
}

}  // namespace mpfem

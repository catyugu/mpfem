#include "mpfem/fem/coefficient.hpp"
#include "mpfem/fem/grid_function.hpp"
#include "mpfem/fem/fe.hpp"
#include <stdexcept>

namespace mpfem {

// ============================================================================
// PWConstCoefficient implementation
// ============================================================================

double PWConstCoefficient::Eval(ElementTransformation& T,
                                 const IntegrationPoint& ip) const {
  (void)ip;  // Not used for piecewise constant
  int attr = T.GetAttribute();
  auto it = values_.find(attr);
  if (it != values_.end()) {
    return it->second;
  }
  return 0.0;
}

// ============================================================================
// FunctionCoefficient implementation
// ============================================================================

double FunctionCoefficient::Eval(ElementTransformation& T,
                                  const IntegrationPoint& ip) const {
  Vec3 p = T.Transform(ip);
  return func_(p);
}

// ============================================================================
// GridFunctionCoefficient implementation
// ============================================================================

double GridFunctionCoefficient::Eval(ElementTransformation& T,
                                      const IntegrationPoint& ip) const {
  if (!gf_) {
    return 0.0;
  }

  // Get the element from transformation
  auto vertices = T.GetElementVertices();
  if (vertices.empty()) {
    return 0.0;
  }

  // Find the element in the mesh
  const Mesh* mesh = gf_->GetFES()->GetMesh();
  const auto& domain_groups = mesh->DomainElements();

  for (const auto& group : domain_groups) {
    for (int e = 0; e < group.Count(); ++e) {
      auto elem_verts = group.GetElementVertices(e);
      if (elem_verts == vertices) {
        return gf_->GetValue(group, e, ip);
      }
    }
  }

  return 0.0;
}

// ============================================================================
// GridFunctionGradientCoefficient implementation
// ============================================================================

void GridFunctionGradientCoefficient::Eval(Vec3& V, ElementTransformation& T,
                                            const IntegrationPoint& ip) const {
  if (!gf_) {
    V.setZero();
    return;
  }

  gf_->GetGradient(T, ip, V);
}

// ============================================================================
// IsotropicElasticityCoefficient implementation
// ============================================================================

void IsotropicElasticityCoefficient::Eval(Mat3& M, ElementTransformation& T,
                                           const IntegrationPoint& ip) const {
  (void)T;
  (void)ip;
  // For simplicity, return identity matrix scaled by E
  // The actual elasticity tensor is 6x6 for 3D
  M.setIdentity();
  M *= E_;
}

Mat6 IsotropicElasticityCoefficient::GetElasticityTensor() const {
  // 3D isotropic elasticity tensor in Voigt notation
  // σ = C : ε
  // C is a 6x6 symmetric matrix

  Mat6 C;
  C.setZero();

  // Plane strain/stress parameters
  double c11 = lambda_ + 2.0 * mu_;
  double c12 = lambda_;
  double c44 = mu_;

  // Fill the matrix
  C(0, 0) = c11;
  C(1, 1) = c11;
  C(2, 2) = c11;

  C(0, 1) = c12;
  C(0, 2) = c12;
  C(1, 0) = c12;
  C(1, 2) = c12;
  C(2, 0) = c12;
  C(2, 1) = c12;

  C(3, 3) = c44;
  C(4, 4) = c44;
  C(5, 5) = c44;

  return C;
}

// ============================================================================
// PWMatrixCoefficient implementation
// ============================================================================

void PWMatrixCoefficient::Eval(Mat3& M, ElementTransformation& T,
                                const IntegrationPoint& ip) const {
  int attr = T.GetAttribute();
  auto it = coefficients_.find(attr);
  if (it != coefficients_.end()) {
    it->second->Eval(M, T, ip);
  } else {
    M.setZero();
  }
}

// ============================================================================
// Helper functions implementation
// ============================================================================

std::unique_ptr<PWConstCoefficient> CreateMaterialPropertyCoefficient(
    const std::map<int, double>& domain_values) {
  auto coeff = std::make_unique<PWConstCoefficient>();
  for (const auto& [domain, value] : domain_values) {
    coeff->SetValue(domain, value);
  }
  return coeff;
}

std::unique_ptr<PWMatrixCoefficient> CreateElasticityCoefficient(
    const std::map<int, std::pair<double, double>>& E_nu_values) {
  auto coeff = std::make_unique<PWMatrixCoefficient>(3);
  for (const auto& [domain, props] : E_nu_values) {
    auto elastic_coeff =
        std::make_shared<IsotropicElasticityCoefficient>(props.first, props.second);
    coeff->SetCoefficient(domain, elastic_coeff);
  }
  return coeff;
}

}  // namespace mpfem

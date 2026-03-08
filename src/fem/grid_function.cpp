#include "mpfem/fem/grid_function.hpp"
#include "mpfem/fem/coefficient.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/core/logger.hpp"
#include <stdexcept>

namespace mpfem {

// ============================================================================
// GridFunction implementation
// ============================================================================

GridFunction::GridFunction(const FiniteElementSpace* fes) { SetSpace(fes); }

void GridFunction::SetSpace(const FiniteElementSpace* fes) {
  fes_ = fes;
  if (fes_) {
    data_.resize(fes_->GetVSize());
    data_.setZero();
  }
}

double GridFunction::GetValue(const ElementGroup& group, int elem_idx,
                               const IntegrationPoint& ip) const {
  // Get element DOFs
  std::vector<int> dofs;
  fes_->GetElementDofs(group, elem_idx, dofs);

  // Get shape functions
  const FiniteElement* fe = fes_->GetFE(group.type);
  std::vector<double> shape(fe->GetDof());
  fe->CalcShape(ip, shape.data());

  // Interpolate
  double value = 0.0;
  int vdim = fes_->GetVDim();

  if (vdim == 1) {
    // Scalar field
    for (size_t i = 0; i < shape.size(); ++i) {
      value += shape[i] * data_(dofs[i]);
    }
  } else {
    // For vector fields, this returns the first component
    for (size_t i = 0; i < shape.size(); ++i) {
      value += shape[i] * data_(dofs[i] * vdim);
    }
  }

  return value;
}

void GridFunction::GetGradient(const ElementGroup& group, int elem_idx,
                               const IntegrationPoint& ip,
                               Vec3& grad) const {
  // Get element DOFs
  std::vector<int> dofs;
  fes_->GetElementDofs(group, elem_idx, dofs);

  // Get shape function gradients in reference coordinates
  const FiniteElement* fe = fes_->GetFE(group.type);
  std::vector<double> dshape(fe->GetDof() * fe->GetDim());
  fe->CalcDShape(ip, dshape.data());

  // Get the mesh and node coordinates
  const Mesh* mesh = fes_->GetMesh();

  // Compute Jacobian and its inverse
  ElementTransformation T;
  T.SetMesh(mesh);
  T.SetElement(&group, elem_idx);

  Mat3 J = T.CalcJacobian(ip);
  Mat3 Jinv = J.inverse();

  // Compute gradient in physical coordinates
  // ∇φ = J^{-T} ∇̂φ
  grad.setZero();

  int vdim = fes_->GetVDim();
  int dof = fe->GetDof();
  int dim = fe->GetDim();

  if (vdim == 1) {
    // Scalar field: gradient is a vector
    Vec3 ref_grad = Vec3::Zero();
    for (int i = 0; i < dof; ++i) {
      for (int j = 0; j < dim; ++j) {
        ref_grad(j) += dshape[i * dim + j] * data_(dofs[i]);
      }
    }
    // Transform to physical coordinates: ∇φ = J^{-T} ∇̂φ
    grad = Jinv.transpose() * ref_grad;
  } else {
    // For vector fields, this is more complex (deformation gradient, etc.)
    // TODO: Implement for vector fields if needed
    throw std::runtime_error(
        "GridFunction::GetGradient for vector fields not yet implemented");
  }
}

void GridFunction::GetGradient(ElementTransformation& T,
                               const IntegrationPoint& ip,
                               Vec3& grad) const {
  // Get the element group and index from transformation
  // This is a convenience method that calls the other GetGradient

  // We need to find the element group - this is a bit awkward
  // For now, assume the transformation has been set up correctly
  const Mesh* mesh = fes_->GetMesh();
  const auto& domain_groups = mesh->DomainElements();

  // Search for the element (this is inefficient but works for now)
  auto vertices = T.GetElementVertices();
  if (vertices.empty()) {
    throw std::runtime_error(
        "GridFunction::GetGradient: transformation not properly set");
  }

  // Find the element in the mesh
  for (const auto& group : domain_groups) {
    for (int e = 0; e < group.Count(); ++e) {
      auto elem_verts = group.GetElementVertices(e);
      if (elem_verts == vertices) {
        GetGradient(group, e, ip, grad);
        return;
      }
    }
  }

  throw std::runtime_error("GridFunction::GetGradient: element not found");
}

void GridFunction::ProjectCoefficient(const Coefficient& coeff) {
  // For H1 order-1 elements, project coefficient at nodes
  const Mesh* mesh = fes_->GetMesh();
  const auto& nodes = mesh->Nodes();

  ElementTransformation T;
  T.SetMesh(mesh);

  int vdim = fes_->GetVDim();

  // For linear elements, simply evaluate coefficient at nodes
  // This is an L2 projection approximation

  // Get domain elements
  const auto& domain_groups = mesh->DomainElements();

  // For each element, project the coefficient
  // This is a simple nodal projection - for higher order elements,
  // a proper L2 projection would be needed

  // Find which elements contain each node
  std::vector<std::vector<std::pair<int, int>>> node_to_elem(
      mesh->NodeCount());  // node -> [(group_idx, elem_idx)]

  for (size_t g = 0; g < domain_groups.size(); ++g) {
    const auto& group = domain_groups[g];
    for (int e = 0; e < group.Count(); ++e) {
      auto verts = group.GetElementVertices(e);
      for (int v : verts) {
        node_to_elem[v].push_back({g, e});
      }
    }
  }

  // Project at each node
  IntegrationPoint ip;  // For linear elements, use centroid

  for (int n = 0; n < mesh->NodeCount(); ++n) {
    if (node_to_elem[n].empty()) continue;

    // Use the first element containing this node
    int g = node_to_elem[n][0].first;
    int e = node_to_elem[n][0].second;

    const auto& group = domain_groups[g];
    T.SetElement(&group, e);

    // For linear elements, the node is at a vertex
    // We need to find which local vertex this is
    auto verts = group.GetElementVertices(e);
    int local_idx = -1;
    for (size_t i = 0; i < verts.size(); ++i) {
      if (verts[i] == n) {
        local_idx = static_cast<int>(i);
        break;
      }
    }

    if (local_idx >= 0) {
      // Get the reference coordinates for this vertex
      const FiniteElement* fe = fes_->GetFE(group.type);

      // For tetrahedron, vertices are at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
      if (group.type == GeometryType::kTetrahedron) {
        double ref_coords[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        ip.x = ref_coords[local_idx][0];
        ip.y = ref_coords[local_idx][1];
        ip.z = ref_coords[local_idx][2];
      } else if (group.type == GeometryType::kTriangle) {
        double ref_coords[3][2] = {{0, 0}, {1, 0}, {0, 1}};
        ip.x = ref_coords[local_idx][0];
        ip.y = ref_coords[local_idx][1];
        ip.z = 0;
      } else {
        // For other element types, use centroid for now
        ip.x = 0.25;
        ip.y = 0.25;
        ip.z = 0.25;
      }

      if (vdim == 1) {
        data_(n) = coeff.Eval(T, ip);
      } else {
        // For vector coefficients, we'd need VectorCoefficient
        // For now, set to zero
        for (int c = 0; c < vdim; ++c) {
          data_(n * vdim + c) = 0.0;
        }
      }
    }
  }
}

}  // namespace mpfem

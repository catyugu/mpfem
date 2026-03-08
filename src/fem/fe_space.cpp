#include "mpfem/fem/fe_space.hpp"
#include "mpfem/core/logger.hpp"
#include <algorithm>
#include <stdexcept>

namespace mpfem {

// ============================================================================
// FiniteElementSpace implementation
// ============================================================================

FiniteElementSpace::FiniteElementSpace(const Mesh* mesh,
                                       const FiniteElementCollection* fec,
                                       int vdim, int ordering)
    : mesh_(mesh),
      fec_(fec),
      vdim_(vdim),
      ordering_(ordering),
      ndofs_(0) {
  BuildDofTable();
}

const FiniteElement* FiniteElementSpace::GetFE(GeometryType geom) const {
  return fec_->GetFiniteElement(geom);
}

void FiniteElementSpace::BuildDofTable() {
  if (!mesh_ || !fec_) {
    throw std::runtime_error("FiniteElementSpace: mesh or fec is null");
  }

  // For H1 order-1 elements, DOFs correspond to nodes
  // For higher order elements, we need to handle edge/face/volume DOFs

  if (fec_->GetKind() == FESpaceKind::kH1 && fec_->GetOrder() == 1) {
    // Simple case: one DOF per node
    ndofs_ = mesh_->NodeCount();

    // For vector-valued fields (e.g., displacement), multiply by vdim
    if (vdim_ > 1) {
      ndofs_ = mesh_->NodeCount() * vdim_;
    }

    BuildDofTableFromMesh();
  } else {
    // Higher order elements: more complex DOF numbering
    // TODO: Implement for higher order elements
    throw std::runtime_error(
        "FiniteElementSpace: Higher order elements not yet implemented");
  }
}

void FiniteElementSpace::BuildDofTableFromMesh() {
  // Build element DOF table for domain elements
  elem_dof_table_.clear();

  const auto& domain_groups = mesh_->DomainElements();
  elem_dof_table_.resize(domain_groups.size());

  for (size_t g = 0; g < domain_groups.size(); ++g) {
    const auto& group = domain_groups[g];
    const FiniteElement* fe = GetFE(group.type);

    // Skip unsupported geometry types
    if (!fe) {
      MPFEM_DEBUG("Skipping domain group %d with unsupported geometry type: %d",
                  static_cast<int>(g), static_cast<int>(group.type));
      continue;
    }

    int dof_per_elem = fe->GetDof();
    int num_elem = group.Count();

    elem_dof_table_[g].resize(num_elem);

    for (int e = 0; e < num_elem; ++e) {
      auto vertices = group.GetElementVertices(e);
      elem_dof_table_[g][e].resize(dof_per_elem * vdim_);

      // For H1 order-1 elements, DOFs are simply the vertex indices
      for (int d = 0; d < dof_per_elem; ++d) {
        int node_idx = vertices[d];
        if (vdim_ == 1) {
          // Scalar field: DOF = node index
          elem_dof_table_[g][e][d] = node_idx;
        } else {
          // Vector field: DOFs for each component
          // Ordering: byNodes = [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...]
          //           byVDim = [u0_x, u1_x, ..., u0_y, u1_y, ..., u0_z, u1_z, ...]
          if (ordering_ == 0) {
            // byNodes ordering
            for (int c = 0; c < vdim_; ++c) {
              elem_dof_table_[g][e][d * vdim_ + c] =
                  node_idx * vdim_ + c;
            }
          } else {
            // byVDim ordering
            for (int c = 0; c < vdim_; ++c) {
              elem_dof_table_[g][e][d + c * dof_per_elem] =
                  node_idx + c * mesh_->NodeCount();
            }
          }
        }
      }
    }
  }

  // Build DOF table for boundary elements
  bdr_elem_dof_table_.clear();

  const auto& bdr_groups = mesh_->BoundaryElements();
  bdr_elem_dof_table_.resize(bdr_groups.size());

  for (size_t g = 0; g < bdr_groups.size(); ++g) {
    const auto& group = bdr_groups[g];
    const FiniteElement* fe = GetFE(group.type);

    // Skip unsupported geometry types (e.g., kPoint)
    if (!fe) {
      MPFEM_DEBUG("Skipping boundary group %d with unsupported geometry type: %d",
                  static_cast<int>(g), static_cast<int>(group.type));
      continue;
    }

    int dof_per_elem = fe->GetDof();
    int num_elem = group.Count();

    bdr_elem_dof_table_[g].resize(num_elem);

    for (int e = 0; e < num_elem; ++e) {
      auto vertices = group.GetElementVertices(e);
      bdr_elem_dof_table_[g][e].resize(dof_per_elem * vdim_);

      for (int d = 0; d < dof_per_elem; ++d) {
        int node_idx = vertices[d];
        if (vdim_ == 1) {
          bdr_elem_dof_table_[g][e][d] = node_idx;
        } else {
          if (ordering_ == 0) {
            for (int c = 0; c < vdim_; ++c) {
              bdr_elem_dof_table_[g][e][d * vdim_ + c] =
                  node_idx * vdim_ + c;
            }
          } else {
            for (int c = 0; c < vdim_; ++c) {
              bdr_elem_dof_table_[g][e][d + c * dof_per_elem] =
                  node_idx + c * mesh_->NodeCount();
            }
          }
        }
      }
    }
  }

  MPFEM_INFO("FiniteElementSpace created: %d DOFs, vdim=%d", ndofs_, vdim_);
}

void FiniteElementSpace::GetElementDofs(int elem_idx,
                                        std::vector<int>& dofs) const {
  // Find which group contains this element
  int count = 0;
  for (size_t g = 0; g < elem_dof_table_.size(); ++g) {
    int num_elem = static_cast<int>(elem_dof_table_[g].size());
    if (elem_idx < count + num_elem) {
      int local_idx = elem_idx - count;
      dofs = elem_dof_table_[g][local_idx];
      return;
    }
    count += num_elem;
  }
  throw std::runtime_error("FiniteElementSpace::GetElementDofs: index out of range");
}

void FiniteElementSpace::GetElementDofs(const ElementGroup& group,
                                        int local_elem_idx,
                                        std::vector<int>& dofs) const {
  // Find the group index
  const auto& domain_groups = mesh_->DomainElements();
  for (size_t g = 0; g < domain_groups.size(); ++g) {
    if (&domain_groups[g] == &group) {
      dofs = elem_dof_table_[g][local_elem_idx];
      return;
    }
  }
  throw std::runtime_error("FiniteElementSpace::GetElementDofs: group not found");
}

void FiniteElementSpace::GetBdrElementDofs(int bdr_elem_idx,
                                           std::vector<int>& dofs) const {
  int count = 0;
  for (size_t g = 0; g < bdr_elem_dof_table_.size(); ++g) {
    int num_elem = static_cast<int>(bdr_elem_dof_table_[g].size());
    if (bdr_elem_idx < count + num_elem) {
      int local_idx = bdr_elem_idx - count;
      dofs = bdr_elem_dof_table_[g][local_idx];
      return;
    }
    count += num_elem;
  }
  throw std::runtime_error("FiniteElementSpace::GetBdrElementDofs: index out of range");
}

void FiniteElementSpace::GetEssentialTrueDofs(const std::vector<int>& bdr_marker,
                                              std::vector<int>& ess_dofs) const {
  // bdr_marker[i] > 0 means boundary attribute i+1 is essential
  ess_dofs.clear();

  const auto& bdr_groups = mesh_->BoundaryElements();

  for (size_t g = 0; g < bdr_groups.size(); ++g) {
    const auto& group = bdr_groups[g];

    // Skip if DOF table is empty (unsupported geometry type)
    if (g >= bdr_elem_dof_table_.size() || bdr_elem_dof_table_[g].empty()) {
      continue;
    }

    for (int e = 0; e < group.Count(); ++e) {
      int bdr_id = group.entity_ids[e];

      // Check if this boundary ID is marked as essential
      if (bdr_id > 0 && bdr_id <= static_cast<int>(bdr_marker.size()) &&
          bdr_marker[bdr_id - 1] > 0) {
        // Get DOFs for this boundary element
        if (e < static_cast<int>(bdr_elem_dof_table_[g].size())) {
          const auto& elem_dofs = bdr_elem_dof_table_[g][e];
          for (int dof : elem_dofs) {
            if (std::find(ess_dofs.begin(), ess_dofs.end(), dof) == ess_dofs.end()) {
              ess_dofs.push_back(dof);
            }
          }
        }
      }
    }
  }

  std::sort(ess_dofs.begin(), ess_dofs.end());
  MPFEM_DEBUG("GetEssentialTrueDofs: %d essential DOFs found", static_cast<int>(ess_dofs.size()));
}

// ============================================================================
// FiniteElementSpaceManager implementation
// ============================================================================

FiniteElementSpace* FiniteElementSpaceManager::CreateSpace(
    const std::string& name, const Mesh* mesh,
    const FiniteElementCollection* fec, int vdim) {
  auto space =
      std::make_unique<FiniteElementSpace>(mesh, fec, vdim);
  FiniteElementSpace* ptr = space.get();
  spaces_[name] = std::move(space);
  return ptr;
}

FiniteElementSpace* FiniteElementSpaceManager::GetSpace(const std::string& name) {
  auto it = spaces_.find(name);
  if (it != spaces_.end()) {
    return it->second.get();
  }
  return nullptr;
}

}  // namespace mpfem

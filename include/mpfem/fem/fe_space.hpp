#pragma once

#include <memory>
#include <vector>

#include "mpfem/core/types.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/mesh/mesh.hpp"

namespace mpfem {

// ============================================================================
// FiniteElementSpace - 有限元空间
// ============================================================================

class FiniteElementSpace {
 public:
  // Constructor for scalar field
  FiniteElementSpace(const Mesh* mesh, const FiniteElementCollection* fec,
                     int vdim = 1, int ordering = 0);

  // Get mesh
  const Mesh* GetMesh() const { return mesh_; }

  // Get finite element collection
  const FiniteElementCollection* GetFEColl() const { return fec_; }

  // Get the finite element for a geometry type
  const FiniteElement* GetFE(GeometryType geom) const;

  // Number of degrees of freedom
  int GetNDofs() const { return ndofs_; }

  // Number of vector components (1 for scalar, 3 for displacement)
  int GetVDim() const { return vdim_; }

  // Total vector size (ndofs_ * vdim_)
  int GetVSize() const { return ndofs_ * vdim_; }

  // Get element DOFs (local to global mapping)
  // Returns indices into the global DOF vector
  void GetElementDofs(int elem_idx, std::vector<int>& dofs) const;

  // Get element DOFs for boundary element
  void GetBdrElementDofs(int bdr_elem_idx, std::vector<int>& dofs) const;

  // Get element DOFs for a specific element group
  void GetElementDofs(const ElementGroup& group, int local_elem_idx,
                      std::vector<int>& dofs) const;

  // Get essential (Dirichlet) boundary DOFs
  void GetEssentialTrueDofs(const std::vector<int>& bdr_marker,
                            std::vector<int>& ess_dofs) const;

  // Build the DOF table (element-to-DOF connectivity)
  void BuildDofTable();

  // Get ordering type (0 = byNodes, 1 = byVDim)
  int GetOrdering() const { return ordering_; }

 private:
  void BuildDofTableFromMesh();

  const Mesh* mesh_;
  const FiniteElementCollection* fec_;
  int vdim_;      // Vector dimension (1 for scalar, 3 for displacement)
  int ordering_;  // 0 = byNodes, 1 = byVDim
  int ndofs_;     // Total number of DOFs

  // Element-to-DOF table: [element_group][element_idx][local_dof] -> global_dof
  std::vector<std::vector<std::vector<int>>> elem_dof_table_;

  // Boundary element-to-DOF table
  std::vector<std::vector<std::vector<int>>> bdr_elem_dof_table_;

  // For H1 elements: DOF = node index
  // For higher order elements, additional DOFs exist
};

// ============================================================================
// FiniteElementSpaceManager - Manage multiple FE spaces
// ============================================================================

class FiniteElementSpaceManager {
 public:
  // Create or get an FE space
  FiniteElementSpace* CreateSpace(const std::string& name, const Mesh* mesh,
                                   const FiniteElementCollection* fec,
                                   int vdim = 1);

  FiniteElementSpace* GetSpace(const std::string& name);

 private:
  std::map<std::string, std::unique_ptr<FiniteElementSpace>> spaces_;
};

}  // namespace mpfem

#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/types.hpp"
#include "fe/fe_space.hpp"
#include "fe/coefficient.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "mesh/mesh.hpp"
#include "solver/sparse_matrix.hpp"
#include <map>
#include <vector>

namespace mpfem {

/// Apply Dirichlet boundary conditions to system matrix and right-hand side vector (using Coefficient)
/// 
/// @param mat System matrix (will be modified: eliminate corresponding rows)
/// @param rhs Right-hand side vector (will be modified: subtract known value contribution)
/// @param sol Solution vector (will be modified: set known values)
/// @param fes Finite element space
/// @param mesh Mesh
/// @param bcValues Boundary condition mapping {boundary ID: Coefficient pointer}
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, const Coefficient*>& bcValues) {
    std::map<Index, Real> dofVals;
    
    // Collect all boundary DOFs and their values
    for (const auto& [bid, coef] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                
                // Get boundary element's reference element and DOF coordinates
                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem) continue;
                
                std::vector<std::vector<Real>> dofCoords = refElem->dofCoords();
                
                // Get boundary element's geometric transform
                FacetElementTransform trans;
                trans.setMesh(&mesh);
                trans.setBoundaryElement(b);
                
                int nd = refElem->numDofs();
                
                // Evaluate Coefficient at each DOF's reference coordinate
                for (int i = 0; i < nd && i < static_cast<int>(dofs.size()); ++i) {
                    Index d = dofs[i];
                    if (d == InvalidIndex || dofVals.find(d) != dofVals.end()) continue;
                    
                    // Get this DOF's coordinate in reference element
                    Real xi[3] = {0.0, 0.0, 0.0};
                    const auto& coord = dofCoords[i];
                    for (size_t c = 0; c < coord.size() && c < 3; ++c) {
                        xi[c] = coord[c];
                    }
                    
                    // Evaluate Coefficient at this DOF's reference coordinate
                    trans.setIntegrationPoint(xi);
                    Real value;
                    if (coef) {
                        coef->eval(trans, value);
                    } else {
                        value = 0.0;
                    }
                    dofVals[d] = value;
                }
            }
        }
    }
    
    mat.eliminateRows(dofVals, rhs);
    for (const auto& [d, v] : dofVals) sol(d) = v;
}

/// Apply Dirichlet boundary conditions to system matrix and right-hand side vector (vector field, using VectorCoefficient)
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, const VectorCoefficient*>& bcValues,
                             int vdim) {
    std::map<Index, Real> dofVals;
    
    for (const auto& [bid, coef] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                
                // Get boundary element's reference element and DOF coordinates
                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem) continue;
                
                std::vector<std::vector<Real>> dofCoords = refElem->dofCoords();
                
                FacetElementTransform trans;
                trans.setMesh(&mesh);
                trans.setBoundaryElement(b);
                
                int nd = refElem->numDofs();
                
                // Evaluate VectorCoefficient at each DOF node's reference coordinate
                for (int i = 0; i < nd; ++i) {
                    // Get this DOF node's coordinate in reference element
                    Real xi[3] = {0.0, 0.0, 0.0};
                    const auto& coord = dofCoords[i];
                    for (size_t c = 0; c < coord.size() && c < 3; ++c) {
                        xi[c] = coord[c];
                    }
                    
                    // Evaluate VectorCoefficient at this DOF node's reference coordinate
                    trans.setIntegrationPoint(xi);
                    Vector3 disp = Vector3::Zero();
                    if (coef) {
                        coef->eval(trans, disp);
                    }
                    
                    // Set all component DOFs for this node
                    for (int c = 0; c < vdim; ++c) {
                        Index d = dofs[i * vdim + c];
                        if (d != InvalidIndex && dofVals.find(d) == dofVals.end()) {
                            dofVals[d] = disp[c];
                        }
                    }
                }
            }
        }
    }
    
    mat.eliminateRows(dofVals, rhs);
    for (const auto& [d, v] : dofVals) sol(d) = v;
}

/// Apply component Dirichlet boundary conditions (single component of vector field)
inline void applyDirichletBCComponent(SparseMatrix& mat, Vector& rhs, Vector& sol,
                                      const FESpace& fes, const Mesh& mesh,
                                      const std::map<int, Real>& componentBCs,
                                      int vdim) {
    std::map<Index, Real> dofVals;
    
    for (const auto& [key, val] : componentBCs) {
        int bid = key / vdim;
        int comp = key % vdim;
        
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                
                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem) continue;
                
                int nd = refElem->numDofs();
                
                for (int i = 0; i < nd; ++i) {
                    Index d = dofs[i * vdim + comp];
                    if (d != InvalidIndex && dofVals.find(d) == dofVals.end()) {
                        dofVals[d] = val;
                    }
                }
            }
        }
    }
    
    mat.eliminateRows(dofVals, rhs);
    for (const auto& [d, v] : dofVals) sol(d) = v;
}

}  // namespace mpfem

#endif  // MPFEM_DIRICHLET_BC_HPP
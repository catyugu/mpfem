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

inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, const Coefficient*>& bcValues) {
    const Index numDofs = fes.numDofs();
    if (numDofs == 0) return;
    
    std::vector<Real> dofVals(numDofs, 0.0);
    std::vector<char> hasVal(numDofs, 0);
    
    FacetElementTransform trans;
    trans.setMesh(&mesh);
    
    for (const auto& [bid, coef] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() != bid) continue;
            
            const ReferenceElement* refElem = fes.bdrElementRefElement(b);
            if (!refElem) continue;
            
            const auto& dofCoords = refElem->dofCoords();
            const int nd = refElem->numDofs();
            const int totalDofs = nd * fes.vdim();
            
            std::vector<Index> dofs(totalDofs);
            fes.getBdrElementDofs(b, std::span<Index>{dofs.data(), static_cast<size_t>(totalDofs)});
            
            trans.setBoundaryElement(b);
            
            for (int i = 0; i < nd && i < static_cast<int>(dofs.size()); ++i) {
                Index d = dofs[i];
                if (d == InvalidIndex || hasVal[d]) continue;
                
                Real xi[3] = {0.0, 0.0, 0.0};
                for (size_t c = 0; c < dofCoords[i].size() && c < 3; ++c) {
                    xi[c] = dofCoords[i][c];
                }
                
                trans.setIntegrationPoint(xi);
                Real value = coef ? 0.0 : 0.0;
                if (coef) coef->eval(trans, value);
                
                dofVals[d] = value;
                hasVal[d] = 1;
            }
        }
    }
    
    std::vector<Index> eliminated;
    eliminated.reserve(numDofs);
    for (Index d = 0; d < numDofs; ++d) {
        if (hasVal[d]) eliminated.push_back(d);
    }
    
    mat.eliminateRows(eliminated, dofVals, rhs);
    for (Index d : eliminated) sol(d) = dofVals[d];
}

inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, const VectorCoefficient*>& bcValues,
                             int vdim) {
    const Index numDofs = fes.numDofs();
    if (numDofs == 0) return;
    
    std::vector<Real> dofVals(numDofs, 0.0);
    std::vector<char> hasVal(numDofs, 0);
    
    FacetElementTransform trans;
    trans.setMesh(&mesh);
    
    for (const auto& [bid, coef] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() != bid) continue;
            
            const ReferenceElement* refElem = fes.bdrElementRefElement(b);
            if (!refElem) continue;
            
            const auto& dofCoords = refElem->dofCoords();
            const int nd = refElem->numDofs();
            const int totalDofs = nd * vdim;
            
            std::vector<Index> dofs(totalDofs);
            fes.getBdrElementDofs(b, std::span<Index>{dofs.data(), static_cast<size_t>(totalDofs)});
            
            trans.setBoundaryElement(b);
            
            for (int i = 0; i < nd; ++i) {
                Real xi[3] = {0.0, 0.0, 0.0};
                for (size_t c = 0; c < dofCoords[i].size() && c < 3; ++c) {
                    xi[c] = dofCoords[i][c];
                }
                
                trans.setIntegrationPoint(xi);
                Vector3 disp = Vector3::Zero();
                if (coef) coef->eval(trans, disp);
                
                for (int c = 0; c < vdim; ++c) {
                    Index d = dofs[i * vdim + c];
                    if (d != InvalidIndex && !hasVal[d]) {
                        dofVals[d] = disp[c];
                        hasVal[d] = 1;
                    }
                }
            }
        }
    }
    
    std::vector<Index> eliminated;
    eliminated.reserve(numDofs);
    for (Index d = 0; d < numDofs; ++d) {
        if (hasVal[d]) eliminated.push_back(d);
    }
    
    mat.eliminateRows(eliminated, dofVals, rhs);
    for (Index d : eliminated) sol(d) = dofVals[d];
}

inline void applyDirichletBCComponent(SparseMatrix& mat, Vector& rhs, Vector& sol,
                                      const FESpace& fes, const Mesh& mesh,
                                      const std::map<int, Real>& componentBCs,
                                      int vdim) {
    const Index numDofs = fes.numDofs();
    if (numDofs == 0) return;
    
    std::vector<Real> dofVals(numDofs, 0.0);
    std::vector<char> hasVal(numDofs, 0);
    
    for (const auto& [key, val] : componentBCs) {
        int bid = key / vdim;
        int comp = key % vdim;
        
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() != bid) continue;
            
            const ReferenceElement* refElem = fes.bdrElementRefElement(b);
            if (!refElem) continue;
            
            const int nd = refElem->numDofs();
            const int totalDofs = nd * vdim;
            
            std::vector<Index> dofs(totalDofs);
            fes.getBdrElementDofs(b, std::span<Index>{dofs.data(), static_cast<size_t>(totalDofs)});
            
            for (int i = 0; i < nd; ++i) {
                Index d = dofs[i * vdim + comp];
                if (d != InvalidIndex && !hasVal[d]) {
                    dofVals[d] = val;
                    hasVal[d] = 1;
                }
            }
        }
    }
    
    std::vector<Index> eliminated;
    eliminated.reserve(numDofs);
    for (Index d = 0; d < numDofs; ++d) {
        if (hasVal[d]) eliminated.push_back(d);
    }
    
    mat.eliminateRows(eliminated, dofVals, rhs);
    for (Index d : eliminated) sol(d) = dofVals[d];
}

}  // namespace mpfem

#endif  // MPFEM_DIRICHLET_BC_HPP
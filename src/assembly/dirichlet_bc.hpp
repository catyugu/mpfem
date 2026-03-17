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

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量（使用 Coefficient）
/// 
/// @param mat 系统矩阵（会被修改：消去对应行）
/// @param rhs 右端向量（会被修改：减去已知值贡献）
/// @param sol 解向量（会被修改：设置已知值）
/// @param fes 有限元空间
/// @param mesh 网格
/// @param bcValues 边界条件映射 {边界ID: Coefficient指针}
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, const Coefficient*>& bcValues) {
    std::map<Index, Real> dofVals;
    
    // 收集所有边界 DOF 及其值
    for (const auto& [bid, coef] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                
                // 获取边界单元的参考单元和DOF坐标
                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem) continue;
                
                std::vector<std::vector<Real>> dofCoords = refElem->dofCoords();
                
                // 获取边界单元的几何变换
                FacetElementTransform trans;
                trans.setMesh(&mesh);
                trans.setBoundaryElement(b);
                
                int nd = refElem->numDofs();
                
                // 对每个DOF在其参考坐标位置评估Coefficient
                for (int i = 0; i < nd && i < static_cast<int>(dofs.size()); ++i) {
                    Index d = dofs[i];
                    if (d == InvalidIndex || dofVals.find(d) != dofVals.end()) continue;
                    
                    // 获取该DOF在参考单元中的坐标
                    Real xi[3] = {0.0, 0.0, 0.0};
                    const auto& coord = dofCoords[i];
                    for (size_t c = 0; c < coord.size() && c < 3; ++c) {
                        xi[c] = coord[c];
                    }
                    
                    // 在该DOF的参考坐标位置评估Coefficient
                    trans.setIntegrationPoint(xi);
                    Real value = coef ? coef->eval(trans) : 0.0;
                    dofVals[d] = value;
                }
            }
        }
    }
    
    mat.eliminateRows(dofVals, rhs);
    for (const auto& [d, v] : dofVals) sol(d) = v;
}

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量（标量场，使用 Real 值）
/// 
/// @deprecated 请使用 Coefficient 版本
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, Real>& bcValues) {
    std::map<Index, Real> dofVals;
    
    // 收集所有边界 DOF 及其值
    for (const auto& [bid, val] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                for (Index d : dofs) {
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

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量（向量场，使用 VectorCoefficient）
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
                
                // 获取边界单元的参考单元和DOF坐标
                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem) continue;
                
                std::vector<std::vector<Real>> dofCoords = refElem->dofCoords();
                
                FacetElementTransform trans;
                trans.setMesh(&mesh);
                trans.setBoundaryElement(b);
                
                int nd = refElem->numDofs();
                
                // 对每个DOF节点在其参考坐标位置评估VectorCoefficient
                for (int i = 0; i < nd; ++i) {
                    // 获取该DOF节点在参考单元中的坐标
                    Real xi[3] = {0.0, 0.0, 0.0};
                    const auto& coord = dofCoords[i];
                    for (size_t c = 0; c < coord.size() && c < 3; ++c) {
                        xi[c] = coord[c];
                    }
                    
                    // 在该DOF节点的参考坐标位置评估VectorCoefficient
                    trans.setIntegrationPoint(xi);
                    Real disp[3] = {0.0, 0.0, 0.0};
                    if (coef) {
                        coef->eval(trans, disp);
                    }
                    
                    // 设置该节点的所有分量DOF
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

/// 应用分量 Dirichlet 边界条件（向量场单个分量）
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

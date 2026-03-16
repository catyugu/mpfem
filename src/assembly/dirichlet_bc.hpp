#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/types.hpp"
#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include "solver/sparse_matrix.hpp"
#include <map>
#include <vector>

namespace mpfem {

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量（标量场）
/// 
/// @param mat 系统矩阵（会被修改：消去对应行）
/// @param rhs 右端向量（会被修改：减去已知值贡献）
/// @param sol 解向量（会被修改：设置已知值）
/// @param fes 有限元空间（用于判断内/外边界）
/// @param mesh 网格
/// @param bcValues 边界条件映射 {边界ID: 值}
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

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量（向量场）
/// 
/// @param mat 系统矩阵
/// @param rhs 右端向量
/// @param sol 解向量
/// @param fes 有限元空间
/// @param mesh 网格
/// @param bcValues 边界条件映射 {边界ID: 位移向量}
/// @param vdim 向量维数（通常为3）
inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
                             const FESpace& fes, const Mesh& mesh,
                             const std::map<int, Vector3>& bcValues,
                             int vdim) {
    std::map<Index, Real> dofVals;
    
    for (const auto& [bid, disp] : bcValues) {
        if (!fes.isExternalBoundaryId(bid)) continue;
        
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                // 向量场：dofs 按 [node0_x, node0_y, node0_z, node1_x, ...] 排列
                const Index nd = dofs.size() / vdim;
                for (Index i = 0; i < nd; ++i) {
                    for (int c = 0; c < vdim; ++c) {
                        Index d = dofs[i * vdim + c];  // 按节点排列
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
                const Index nd = dofs.size() / vdim;
                for (Index i = 0; i < nd; ++i) {
                    Index d = dofs[i * vdim + comp];  // 按节点排列
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

#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/types.hpp"
#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include "solver/sparse_matrix.hpp"
#include <map>
#include <vector>

namespace mpfem {

/// 应用 Dirichlet 边界条件到系统矩阵和右端向量
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
    // 注意：只对外边界应用边界条件，跳过内边界
    for (const auto& [bid, val] : bcValues) {
        for (Index b = 0; b < mesh.numBdrElements(); ++b) {
            // 跳过内边界 - 只对外边界应用 Dirichlet BC
            if (!fes.isExternalBoundary(b)) {
                continue;
            }
            
            if (mesh.bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes.getBdrElementDofs(b, dofs);
                for (Index d : dofs) {
                    // 第一个设置的值生效（处理角点）
                    if (d != InvalidIndex && dofVals.find(d) == dofVals.end()) {
                        dofVals[d] = val;
                    }
                }
            }
        }
    }
    
    // 消去矩阵行并修正右端向量
    mat.eliminateRows(dofVals, rhs);
    
    // 设置解向量的已知值
    for (const auto& [d, v] : dofVals) {
        sol(d) = v;
    }
}

}  // namespace mpfem

#endif  // MPFEM_DIRICHLET_BC_HPP

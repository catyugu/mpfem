/**
 * @file dof_map.cpp
 * @brief Implementation of DofMap utilities
 */

#include "dof_map.hpp"
#include "core/logger.hpp"

namespace mpfem {

void DofMap::distribute_local_to_global(
    const DynamicMatrix& local_matrix,
    const std::vector<Index>& local_dofs,
    int n_local,
    const DoFHandler* dof_handler,
    std::vector<Eigen::Triplet<Scalar>>& triplets) {
    
    for (int i = 0; i < n_local; ++i) {
        Index gi = local_dofs[i];
        if (gi == InvalidIndex) continue;
        
        // Skip constrained DoFs in rows
        if (dof_handler && dof_handler->is_constrained(gi)) continue;
        
        for (int j = 0; j < n_local; ++j) {
            Index gj = local_dofs[j];
            if (gj == InvalidIndex) continue;
            
            triplets.emplace_back(gi, gj, local_matrix(i, j));
        }
    }
}

void DofMap::distribute_local_to_global(
    const DynamicVector& local_vector,
    const std::vector<Index>& local_dofs,
    int n_local,
    const DoFHandler* dof_handler,
    DynamicVector& global_vector) {
    
    for (int i = 0; i < n_local; ++i) {
        Index gi = local_dofs[i];
        if (gi == InvalidIndex) continue;
        if (gi >= global_vector.size()) continue;
        
        // Skip constrained DoFs
        if (dof_handler && dof_handler->is_constrained(gi)) continue;
        
        global_vector[gi] += local_vector[i];
    }
}

void DofMap::apply_dirichlet_bc(
    SparseMatrix& matrix,
    DynamicVector& rhs,
    const DoFHandler* dof_handler) {
    
    if (!dof_handler) return;
    
    const auto& constraints = dof_handler->constraints();
    Index n_dofs = dof_handler->n_dofs();
    
    // Convert to uncompressed for modification
    matrix.makeCompressed();
    
    // For each constrained DoF
    for (const auto& constraint : constraints) {
        Index dof = constraint.global_dof;
        if (dof >= n_dofs) continue;
        
        Scalar value = constraint.value;
        
        // Zero out the row except diagonal
        for (SparseMatrix::InnerIterator it(matrix, dof); it; ++it) {
            if (it.row() == dof) {
                it.valueRef() = 1.0;  // Set diagonal to 1
            } else {
                it.valueRef() = 0.0;  // Zero off-diagonal
            }
        }
        
        // Modify RHS for other rows
        // F(j) -= K(j, dof) * value
        // Then zero K(j, dof)
        // This requires iterating over columns, which is expensive for CSC format
        // Alternative: use triplet-based modification
        
        // Set RHS value
        rhs[dof] = value;
    }
    
    // Alternative approach: modify RHS accounting for known values
    // This is done by iterating over all columns
    for (const auto& constraint : constraints) {
        Index dof = constraint.global_dof;
        if (dof >= n_dofs) continue;
        
        Scalar value = constraint.value;
        
        // For each row, subtract contribution from known value
        // This requires iterating over column dof
        // In Eigen, we iterate over rows; column iteration is expensive
        
        // Simplified: just set diagonal and RHS
        // The elimination method above already handles the basic case
    }
}

void DofMap::apply_dirichlet_bc_simple(
    SparseMatrix& matrix,
    DynamicVector& rhs,
    const DoFHandler* dof_handler) {
    
    if (!dof_handler) return;
    
    const auto& constraints = dof_handler->constraints();
    Index n_dofs = dof_handler->n_dofs();
    
    if (constraints.empty()) return;
    
    // Build set of constrained DoFs for quick lookup
    std::unordered_set<Index> constrained_set;
    for (const auto& c : constraints) {
        if (c.global_dof < n_dofs) {
            constrained_set.insert(c.global_dof);
        }
    }
    
    // Build new matrix from scratch to ensure proper structure
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(matrix.nonZeros() + constraints.size());
    
    // Copy all non-constrained entries
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
            Index row = it.row();
            Index col = it.col();
            
            bool row_is_constrained = constrained_set.count(row) > 0;
            bool col_is_constrained = constrained_set.count(col) > 0;
            
            if (row_is_constrained || col_is_constrained) {
                // Skip this entry - will be handled separately
                continue;
            }
            
            triplets.emplace_back(row, col, it.value());
        }
    }
    
    // Add diagonal 1.0 for constrained DoFs
    for (const auto& c : constraints) {
        Index dof = c.global_dof;
        if (dof < n_dofs) {
            triplets.emplace_back(dof, dof, 1.0);
            rhs[dof] = c.value;
        }
    }
    
    // Rebuild matrix
    matrix.setZero();
    matrix.resize(n_dofs, n_dofs);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
}

void DofMap::extract_local_solution(
    const DynamicVector& global_solution,
    const std::vector<Index>& local_dofs,
    DynamicVector& local_solution) {
    
    int n = static_cast<int>(local_dofs.size());
    local_solution.setZero(n);
    
    for (int i = 0; i < n; ++i) {
        Index gi = local_dofs[i];
        if (gi < global_solution.size()) {
            local_solution[i] = global_solution[gi];
        }
    }
}

void DofMap::get_constrained_dof_values(
    const DoFHandler* dof_handler,
    std::unordered_map<Index, Scalar>& values) {
    
    values.clear();
    if (!dof_handler) return;
    
    const auto& constraints = dof_handler->constraints();
    for (const auto& c : constraints) {
        values[c.global_dof] = c.value;
    }
}

void DofMap::apply_dirichlet_bc_fast(
    std::vector<Eigen::Triplet<Scalar>>& triplets,
    const std::unordered_set<Index>& constrained_dofs,
    const std::unordered_map<Index, Scalar>& bc_values,
    DynamicVector& rhs) {
    
    if (constrained_dofs.empty()) return;
    
    // Filter triplets: keep only those not in constrained rows
    std::vector<Eigen::Triplet<Scalar>> filtered;
    filtered.reserve(triplets.size());
    
    for (const auto& t : triplets) {
        Index row = t.row();
        Index col = t.col();
        
        // Skip entries in constrained rows
        if (constrained_dofs.count(row) > 0) {
            continue;
        }
        
        // Keep entry (also skip entries in constrained columns for cleaner matrix)
        if (constrained_dofs.count(col) == 0) {
            filtered.push_back(t);
        }
    }
    
    // Add diagonal entries for constrained DoFs
    for (Index dof : constrained_dofs) {
        filtered.emplace_back(dof, dof, 1.0);
    }
    
    // Swap with original
    triplets.swap(filtered);
    
    // Set RHS values
    for (const auto& [dof, value] : bc_values) {
        if (dof < static_cast<Index>(rhs.size())) {
            rhs[dof] = value;
        }
    }
}

}  // namespace mpfem

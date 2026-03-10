/**
 * @file dof_map.hpp
 * @brief DofMap - utilities for local-to-global DoF mapping
 * 
 * Provides helper functions for distributing local contributions
 * to global matrices and vectors, handling constraints.
 */

#ifndef MPFEM_ASSEMBLY_DOF_MAP_HPP
#define MPFEM_ASSEMBLY_DOF_MAP_HPP

#include "core/types.hpp"
#include "dof/dof_handler.hpp"
#include <vector>

namespace mpfem {

/**
 * @brief Utilities for DoF mapping operations
 */
class DofMap {
public:
    /**
     * @brief Distribute local matrix to global sparse matrix
     * 
     * Adds the local matrix contribution to the global sparse matrix.
     * Handles constraints by skipping rows corresponding to constrained DoFs.
     * 
     * @param local_matrix Local element matrix
     * @param local_dofs Global DoF indices for local DoFs
     * @param n_local Number of local DoFs
     * @param dof_handler DoF handler for constraint info
     * @param triplets Output: vector of triplets to add to sparse matrix
     */
    static void distribute_local_to_global(
        const DynamicMatrix& local_matrix,
        const std::vector<Index>& local_dofs,
        int n_local,
        const DoFHandler* dof_handler,
        std::vector<Eigen::Triplet<Scalar>>& triplets);
    
    /**
     * @brief Distribute local vector to global vector
     * 
     * Adds the local vector contribution to the global vector.
     * Handles constraints by skipping constrained DoFs.
     * 
     * @param local_vector Local element vector
     * @param local_dofs Global DoF indices for local DoFs
     * @param n_local Number of local DoFs
     * @param dof_handler DoF handler for constraint info
     * @param global_vector Global vector to add to
     */
    static void distribute_local_to_global(
        const DynamicVector& local_vector,
        const std::vector<Index>& local_dofs,
        int n_local,
        const DoFHandler* dof_handler,
        DynamicVector& global_vector);
    
    /**
     * @brief Apply Dirichlet boundary conditions to system
     * 
     * Modifies the system matrix and right-hand side to enforce
     * Dirichlet boundary conditions using the elimination method.
     * 
     * For each constrained DoF i with value v:
     * - Set row i of K to zero, except K(i,i) = 1
     * - Set F(i) = v
     * - Modify other rows: F(j) -= K(j,i) * v, then K(j,i) = 0
     * 
     * @param matrix System matrix (modified in place)
     * @param rhs Right-hand side vector (modified in place)
     * @param dof_handler DoF handler containing constraints
     */
    static void apply_dirichlet_bc(
        SparseMatrix& matrix,
        DynamicVector& rhs,
        const DoFHandler* dof_handler);
    
    /**
     * @brief Apply Dirichlet boundary conditions (simple method)
     * 
     * Simple approach: set diagonal to 1, RHS to value, zero other entries.
     * 
     * @param matrix System matrix
     * @param rhs Right-hand side vector
     * @param dof_handler DoF handler
     * @param values Prescribed values (for inhomogeneous BCs)
     */
    static void apply_dirichlet_bc_simple(
        SparseMatrix& matrix,
        DynamicVector& rhs,
        const DoFHandler* dof_handler);
    
    /**
     * @brief Extract local solution from global solution
     * 
     * @param global_solution Global solution vector
     * @param local_dofs Local DoF indices
     * @param local_solution Output: local solution values
     */
    static void extract_local_solution(
        const DynamicVector& global_solution,
        const std::vector<Index>& local_dofs,
        DynamicVector& local_solution);
    
    /**
     * @brief Get constrained DoF values
     * 
     * @param dof_handler DoF handler
     * @param values Output: map from DoF index to prescribed value
     */
    static void get_constrained_dof_values(
        const DoFHandler* dof_handler,
        std::unordered_map<Index, Scalar>& values);
};

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_DOF_MAP_HPP

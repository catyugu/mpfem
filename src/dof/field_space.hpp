/**
 * @file field_space.hpp
 * @brief Field space - manages DoF mapping for a single physical field
 * 
 * This class separates the concept of "physical field" from DoF numbering.
 * Each field has:
 * - A unique name (e.g., "electric_potential", "temperature", "displacement")
 * - Number of components (1 for scalar, 2/3 for vector)
 * - Its own solution vector
 * - Mapping from field-local DoF to global DoF
 */

#ifndef MPFEM_DOF_FIELD_SPACE_HPP
#define MPFEM_DOF_FIELD_SPACE_HPP

#include "core/types.hpp"
#include "dof_table.hpp"
#include "mesh/mesh.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace mpfem {

/**
 * @brief Field identifier - unique name for a physical field
 */
using FieldID = std::string;

/**
 * @brief Field type enumeration
 */
enum class FieldType {
    Scalar,   // Single component (temperature, pressure, electric potential)
    Vector    // Multiple components (displacement, velocity)
};

/**
 * @brief Manages DoF mapping and solution for a single physical field
 * 
 * Key concept: The FieldSpace owns the solution vector and provides
 * name-based access to field values, hiding the underlying DoF indexing.
 * 
 * Usage:
 * @code
 * FieldSpace temperature("temperature", mesh, 1);  // Scalar field
 * FieldSpace displacement("displacement", mesh, 3); // 3D vector field
 * 
 * // Set boundary condition
 * temperature.set_bc(boundary_id, 293.15);
 * 
 * // Access solution at node
 * double T = temperature.value_at_node(node_id);
 * 
 * // For vector fields
 * Vector3 u = displacement.vector_at_node(node_id);
 * @endcode
 */
class FieldSpace {
public:
    /**
     * @brief Construct a field space
     * @param name Unique field name
     * @param mesh The mesh
     * @param n_components Number of components (1 for scalar, dim for vector)
     * @param order Polynomial order (1 for linear, 2 for quadratic)
     */
    FieldSpace(const FieldID& name, const Mesh* mesh, int n_components, int order = 1);
    
    ~FieldSpace() = default;
    
    // ============================================================
    // Field Properties
    // ============================================================
    
    /// Get field name
    const FieldID& name() const { return name_; }
    
    /// Get field type
    FieldType type() const { 
        return n_components_ == 1 ? FieldType::Scalar : FieldType::Vector; 
    }
    
    /// Number of components
    int n_components() const { return n_components_; }
    
    /// Polynomial order
    int order() const { return order_; }
    
    /// Total number of DoFs for this field
    Index n_dofs() const { return n_dofs_; }
    
    /// Number of nodes (vertices for Lagrange elements)
    SizeType n_nodes() const { return mesh_->num_vertices(); }
    
    // ============================================================
    // Solution Vector Access
    // ============================================================
    
    /// Get the solution vector
    DynamicVector& solution() { return solution_; }
    const DynamicVector& solution() const { return solution_; }
    
    /// Get scalar value at a node (for scalar fields)
    Scalar value_at_node(Index node_id) const {
        if (n_components_ != 1) return 0.0;
        Index dof = node_dof(node_id, 0);
        return (dof >= 0 && dof < solution_.size()) ? solution_[dof] : 0.0;
    }
    
    /// Set scalar value at a node
    void set_value_at_node(Index node_id, Scalar value) {
        if (n_components_ != 1) return;
        Index dof = node_dof(node_id, 0);
        if (dof >= 0 && dof < solution_.size()) {
            solution_[dof] = value;
        }
    }
    
    /// Get vector value at a node (for vector fields)
    Tensor<1, 3> vector_at_node(Index node_id) const {
        Tensor<1, 3> result = Tensor<1, 3>::Zero();
        for (int c = 0; c < n_components_ && c < 3; ++c) {
            Index dof = node_dof(node_id, c);
            if (dof >= 0 && dof < solution_.size()) {
                result[c] = solution_[dof];
            }
        }
        return result;
    }
    
    /// Set vector value at a node
    void set_vector_at_node(Index node_id, const Tensor<1, 3>& value) {
        for (int c = 0; c < n_components_ && c < 3; ++c) {
            Index dof = node_dof(node_id, c);
            if (dof >= 0 && dof < solution_.size()) {
                solution_[dof] = value[c];
            }
        }
    }
    
    /// Get DoF index for a node and component
    Index node_dof(Index node_id, int component = 0) const {
        if (component < 0 || component >= n_components_) return InvalidIndex;
        return node_id * n_components_ + component;
    }
    
    /// Get all DoF indices for a node
    std::vector<Index> node_dofs(Index node_id) const {
        std::vector<Index> dofs(n_components_);
        for (int c = 0; c < n_components_; ++c) {
            dofs[c] = node_dof(node_id, c);
        }
        return dofs;
    }
    
    // ============================================================
    // Cell DoF Access
    // ============================================================
    
    /**
     * @brief Get DoF indices for a cell
     * @param cell_id Global cell index
     * @param dofs Output vector of DoF indices
     */
    void get_cell_dofs(Index cell_id, std::vector<Index>& dofs) const;
    
    /**
     * @brief Get number of DoFs per cell
     */
    int dofs_per_cell() const;
    
    // ============================================================
    // Boundary Conditions
    // ============================================================
    
    /**
     * @brief Add Dirichlet boundary condition
     * @param boundary_id Boundary entity ID
     * @param value Prescribed value (for scalar) or component value (for vector)
     * @param component Component index (-1 for all components)
     */
    void add_dirichlet_bc(Index boundary_id, Scalar value, int component = -1);
    
    /**
     * @brief Get Dirichlet BC constraints
     */
    const std::vector<std::pair<Index, Scalar>>& dirichlet_constraints() const {
        return dirichlet_bcs_;
    }
    
    /**
     * @brief Check if a DoF is constrained
     */
    bool is_constrained(Index dof) const {
        return constrained_dofs_.count(dof) > 0;
    }
    
    /**
     * @brief Apply boundary conditions to system
     */
    void apply_bcs_to_system(SparseMatrix& K, DynamicVector& f) const;
    
    // ============================================================
    // Initialization
    // ============================================================
    
    /**
     * @brief Initialize solution vector (resize and set to zero)
     */
    void initialize_solution() {
        solution_.setZero(n_dofs_);
    }
    
    /**
     * @brief Set solution from external vector
     */
    void set_solution(const DynamicVector& sol) {
        solution_ = sol;
    }
    
private:
    FieldID name_;
    const Mesh* mesh_;
    int n_components_;
    int order_;
    Index n_dofs_;
    
    DynamicVector solution_;
    
    // Boundary conditions
    std::vector<std::pair<Index, Scalar>> dirichlet_bcs_;
    std::unordered_set<Index> constrained_dofs_;
    
    // Cell DoF table
    DoFTable cell_dof_table_;
    
    void build_cell_dof_table();
};

/**
 * @brief Shared pointer to FieldSpace
 */
using FieldSpacePtr = std::shared_ptr<FieldSpace>;

} // namespace mpfem

#endif // MPFEM_DOF_FIELD_SPACE_HPP

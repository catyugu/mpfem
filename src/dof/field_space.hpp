/**
 * @file field_space.hpp
 * @brief Field space - manages DoF mapping for a single physical field
 * 
 * DESIGN PRINCIPLE:
 * =================
 * This class provides TWO levels of API:
 * 
 * 1. HIGH-LEVEL API (for physics assembly users):
 *    - value_at_node(), vector_at_node() - access by node ID
 *    - add_dirichlet_bc() - by boundary ID
 *    - Users should NOT need to call get_cell_dofs() directly
 * 
 * 2. LOW-LEVEL API (for framework internals like FEValues):
 *    - get_cell_dofs(), node_dof() - DoF index access
 *    - These are public but marked as "internal use"
 * 
 * The philosophy: DoF indices are hidden from END USERS, but 
 * framework components naturally need access to them.
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

using FieldID = std::string;

enum class FieldType {
    Scalar,   // temperature, pressure, electric potential
    Vector    // displacement, velocity
};

/**
 * @brief Manages DoF mapping and solution for a single physical field
 */
class FieldSpace {
public:
    FieldSpace(const FieldID& name, const Mesh* mesh, int n_components, int order = 1);
    ~FieldSpace() = default;
    
    // ============================================================
    // Field Properties
    // ============================================================
    
    const FieldID& name() const { return name_; }
    FieldType type() const { return n_components_ == 1 ? FieldType::Scalar : FieldType::Vector; }
    int n_components() const { return n_components_; }
    int order() const { return order_; }
    Index n_dofs() const { return n_dofs_; }
    SizeType n_nodes() const { return mesh_->num_vertices(); }
    const Mesh* mesh() const { return mesh_; }
    
    // ============================================================
    // HIGH-LEVEL API: Solution Value Access (by Node ID)
    // ============================================================
    
    Scalar value_at_node(Index node_id) const;
    void set_value_at_node(Index node_id, Scalar value);
    
    Tensor<1, 3> vector_at_node(Index node_id) const;
    void set_vector_at_node(Index node_id, const Tensor<1, 3>& value);
    
    Scalar component_at_node(Index node_id, int component) const;
    void set_component_at_node(Index node_id, int component, Scalar value);
    
    // ============================================================
    // Solution Vector Access
    // ============================================================
    
    DynamicVector& solution() { return solution_; }
    const DynamicVector& solution() const { return solution_; }
    
    void initialize_solution() { solution_.setZero(n_dofs_); }
    void set_solution(const DynamicVector& sol) { solution_ = sol; }
    
    // ============================================================
    // HIGH-LEVEL API: Boundary Conditions (by Boundary ID)
    // ============================================================
    
    void add_dirichlet_bc(Index boundary_id, Scalar value, int component = -1);
    void apply_bcs_to_system(SparseMatrix& K, DynamicVector& f) const;
    
    bool is_node_constrained(Index node_id) const;
    Index n_constrained_nodes() const;
    
    // ============================================================
    // System Info
    // ============================================================
    
    Index n_free_dofs() const { return n_dofs_ - static_cast<Index>(constrained_dofs_.size()); }
    Index n_constrained_dofs() const { return static_cast<Index>(constrained_dofs_.size()); }
    
    // ============================================================
    // LOW-LEVEL API: DoF Index Access (INTERNAL USE)
    // Used by FEValues, FieldRegistry, and other framework components
    // ============================================================
    
    /// @internal Get DoF index for a node and component
    Index node_dof(Index node_id, int component = 0) const {
        if (component < 0 || component >= n_components_) return InvalidIndex;
        return node_id * n_components_ + component;
    }
    
    /// @internal Get all DoF indices for a node
    std::vector<Index> node_dofs(Index node_id) const {
        std::vector<Index> dofs(n_components_);
        for (int c = 0; c < n_components_; ++c) {
            dofs[c] = node_dof(node_id, c);
        }
        return dofs;
    }
    
    /// @internal Get DoF indices for a cell
    void get_cell_dofs(Index cell_id, std::vector<Index>& dofs) const;
    
    /// @internal Get number of DoFs per cell
    int dofs_per_cell() const;
    
    /// @internal Check if a DoF is constrained
    bool is_constrained(Index dof) const {
        return constrained_dofs_.count(dof) > 0;
    }
    
    /// @internal Get constraint value for a DoF
    Scalar get_constraint_value(Index dof) const {
        for (const auto& [d, v] : dirichlet_bcs_) {
            if (d == dof) return v;
        }
        return 0.0;
    }

private:
    FieldID name_;
    const Mesh* mesh_;
    int n_components_;
    int order_;
    Index n_dofs_;
    
    DynamicVector solution_;
    
    std::vector<std::pair<Index, Scalar>> dirichlet_bcs_;
    std::unordered_set<Index> constrained_dofs_;
    
    DoFTable cell_dof_table_;
    
    void build_cell_dof_table();
};

using FieldSpacePtr = std::shared_ptr<FieldSpace>;

} // namespace mpfem

#endif // MPFEM_DOF_FIELD_SPACE_HPP
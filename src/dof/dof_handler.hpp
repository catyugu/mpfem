/**
 * @file dof_handler.hpp
 * @brief DoF handler - manages distribution of degrees of freedom
 */

#ifndef MPFEM_DOF_DOF_HANDLER_HPP
#define MPFEM_DOF_DOF_HANDLER_HPP

#include "fe_space.hpp"
#include "dof_table.hpp"
#include "mesh/mesh.hpp"
#include "core/types.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace mpfem {

/**
 * @brief Constraint information for a DoF
 */
struct DoFConstraint {
    Index global_dof;
    Scalar value;

    DoFConstraint() : global_dof(InvalidIndex), value(0) {}
    DoFConstraint(Index dof, Scalar val) : global_dof(dof), value(val) {}
};

/**
 * @brief Manages DoF distribution and constraints
 */
class DoFHandler {
public:
    DoFHandler();

    /**
     * @brief Initialize with FE space
     */
    void initialize(const FESpace* fe_space);

    /**
     * @brief Distribute DoFs
     */
    void distribute_dofs();

    /**
     * @brief Add Dirichlet BC on a boundary
     * @param boundary_id Boundary ID
     * @param value Prescribed value
     * @param component Component index (-1 for all components)
     */
    void add_dirichlet_bc(Index boundary_id, Scalar value, int component = -1);

    /**
     * @brief Apply boundary conditions
     */
    void apply_boundary_conditions();

    // Accessors
    Index n_dofs() const { return n_dofs_; }
    Index n_constrained_dofs() const { return static_cast<Index>(constraints_.size()); }
    Index n_free_dofs() const { return n_dofs_ - n_constrained_dofs(); }

    bool is_constrained(Index global_dof) const {
        return constrained_dofs_.count(global_dof) > 0;
    }

    const DoFConstraint* get_constraint(Index global_dof) const;
    const std::vector<DoFConstraint>& constraints() const { return constraints_; }

    const DoFTable& dof_table() const { return dof_table_; }

    void get_cell_dofs(Index global_cell_idx, std::vector<Index>& dofs) const;

    Index free_to_global(Index free_dof) const {
        return free_dof < static_cast<Index>(free_dofs_.size())
               ? free_dofs_[free_dof] : InvalidIndex;
    }

    Index global_to_free(Index global_dof) const;

    const FESpace* fe_space() const { return fe_space_; }

    void clear();

private:
    const FESpace* fe_space_;
    const Mesh* mesh_;

    Index n_dofs_;
    DoFTable dof_table_;

    std::vector<DoFConstraint> constraints_;
    std::unordered_set<Index> constrained_dofs_;

    std::vector<Index> global_to_free_;
    std::vector<Index> free_dofs_;

    struct BCInfo {
        Index boundary_id;
        Scalar value;
        int component;
    };
    std::vector<BCInfo> pending_bcs_;

    void build_dof_table();
    void process_boundary_conditions();
};

} // namespace mpfem

#endif // MPFEM_DOF_DOF_HANDLER_HPP

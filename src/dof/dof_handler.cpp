/**
 * @file dof_handler.cpp
 * @brief Implementation of DoF handler
 */

#include "dof_handler.hpp"
#include "mesh/mesh.hpp"
#include "mesh/element.hpp"
#include "mesh/geometry.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
#include <algorithm>

namespace mpfem {

DoFHandler::DoFHandler()
    : fe_space_(nullptr)
    , mesh_(nullptr)
    , n_dofs_(0)
{
}

void DoFHandler::initialize(const FESpace* fe_space) {
    if (!fe_space) {
        MPFEM_THROW(InvalidArgument, "FE space pointer is null");
    }

    fe_space_ = fe_space;
    mesh_ = fe_space->mesh();

    if (!mesh_) {
        MPFEM_THROW(InvalidArgument, "Mesh pointer in FE space is null");
    }

    clear();
    MPFEM_INFO("DoFHandler initialized");
}

void DoFHandler::distribute_dofs() {
    if (!fe_space_) {
        MPFEM_THROW(RuntimeError, "FE space not set");
    }

    build_dof_table();
    n_dofs_ = fe_space_->n_dofs();

    global_to_free_.resize(n_dofs_);
    free_dofs_.reserve(n_dofs_);

    for (Index i = 0; i < n_dofs_; ++i) {
        global_to_free_[i] = i;
        free_dofs_.push_back(i);
    }

    MPFEM_INFO("Distributed " << n_dofs_ << " DoFs");
}

void DoFHandler::add_dirichlet_bc(Index boundary_id, Scalar value, int component) {
    pending_bcs_.push_back({boundary_id, value, component});
}

void DoFHandler::apply_boundary_conditions() {
    process_boundary_conditions();

    free_dofs_.clear();
    for (Index i = 0; i < n_dofs_; ++i) {
        if (!is_constrained(i)) {
            global_to_free_[i] = static_cast<Index>(free_dofs_.size());
            free_dofs_.push_back(i);
        } else {
            global_to_free_[i] = InvalidIndex;
        }
    }

    MPFEM_INFO("Applied BCs: " << n_constrained_dofs()
               << " constrained, " << n_free_dofs() << " free");
}

const DoFConstraint* DoFHandler::get_constraint(Index global_dof) const {
    for (const auto& c : constraints_) {
        if (c.global_dof == global_dof) {
            return &c;
        }
    }
    return nullptr;
}

Index DoFHandler::global_to_free(Index global_dof) const {
    if (global_dof < 0 || global_dof >= static_cast<Index>(global_to_free_.size())) {
        return InvalidIndex;
    }
    return global_to_free_[global_dof];
}

void DoFHandler::get_cell_dofs(Index global_cell_idx, std::vector<Index>& dofs) const {
    dofs.clear();

    if (global_cell_idx < 0 ||
        static_cast<SizeType>(global_cell_idx) >= dof_table_.n_cells()) {
        return;
    }

    dof_table_.get_cell_dofs(static_cast<SizeType>(global_cell_idx), dofs);
}

void DoFHandler::clear() {
    n_dofs_ = 0;
    dof_table_.clear();
    constraints_.clear();
    constrained_dofs_.clear();
    global_to_free_.clear();
    free_dofs_.clear();
    pending_bcs_.clear();
}

void DoFHandler::build_dof_table() {
    SizeType n_cells = mesh_->num_cells();
    const auto& blocks = mesh_->cell_blocks();

    // Count dofs per cell
    std::vector<int> dp_cell;
    dp_cell.reserve(n_cells);

    for (const auto& block : blocks) {
        GeometryType geom_type = to_geometry_type(block.type());
        int dp = fe_space_->dofs_per_cell(geom_type);
        for (SizeType i = 0; i < block.size(); ++i) {
            dp_cell.push_back(dp);
        }
    }

    dof_table_ = DoFTable(dp_cell);

    // Fill dof indices
    SizeType cell_idx = 0;
    std::vector<Index> cell_dofs;

    for (SizeType b = 0; b < blocks.size(); ++b) {
        const auto& block = blocks[b];
        for (SizeType i = 0; i < block.size(); ++i, ++cell_idx) {
            fe_space_->get_cell_dofs(b, i, cell_dofs);
            dof_table_.set_cell_dofs(cell_idx, cell_dofs);
        }
    }

    MPFEM_INFO("Built DoF table: " << n_cells << " cells, "
               << dof_table_.total_entries() << " entries");
}

void DoFHandler::process_boundary_conditions() {
    const auto& geom = mesh_->geometry();
    int n_comp = fe_space_->n_components();

    // Build boundary vertex map using GeometryManager
    std::unordered_map<Index, std::unordered_set<Index>> boundary_vertices;

    const auto& face_blocks = mesh_->face_blocks();
    Index global_face_idx = 0;

    for (const auto& block : face_blocks) {
        for (SizeType i = 0; i < block.size(); ++i, ++global_face_idx) {
            Index bnd_id = block.entity_id(i);
            auto verts = block.element_vertices(i);
            for (Index v : verts) {
                boundary_vertices[bnd_id].insert(v);
            }
        }
    }

    // Apply pending BCs
    for (const auto& bc : pending_bcs_) {
        auto it = boundary_vertices.find(bc.boundary_id);
        if (it == boundary_vertices.end()) {
            MPFEM_WARN("Boundary " << bc.boundary_id << " not found");
            continue;
        }

        for (Index vertex_id : it->second) {
            if (bc.component < 0) {
                // All components
                for (int c = 0; c < n_comp; ++c) {
                    Index dof = vertex_id * n_comp + c;
                    if (constrained_dofs_.insert(dof).second) {
                        constraints_.emplace_back(dof, bc.value);
                    }
                }
            } else if (bc.component < n_comp) {
                Index dof = vertex_id * n_comp + bc.component;
                if (constrained_dofs_.insert(dof).second) {
                    constraints_.emplace_back(dof, bc.value);
                }
            }
        }
    }

    pending_bcs_.clear();
}

} // namespace mpfem
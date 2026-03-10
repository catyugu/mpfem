/**
 * @file fe_space.cpp
 * @brief Implementation of finite element space
 */

#include "fe_space.hpp"
#include "mesh/element.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

FESpace::FESpace(const Mesh* mesh, const std::string& fe_name, int n_components)
    : mesh_(mesh)
    , n_components_(n_components)
    , n_dofs_(0)
{
    if (!mesh_) {
        MPFEM_THROW(InvalidArgument, "Mesh pointer is null");
    }

    // Create FE for each geometry type present in mesh
    const auto& cell_blocks = mesh_->cell_blocks();

    for (const auto& block : cell_blocks) {
        GeometryType geom_type = to_geometry_type(block.type());

        if (fe_by_geom_.find(geom_type) == fe_by_geom_.end()) {
            auto fe = FECollection::create(fe_name, geom_type, n_components_);
            if (fe) {
                fe_by_geom_[geom_type] = std::move(fe);
            }
        }
    }

    MPFEM_INFO("Created FESpace with " << fe_by_geom_.size()
               << " geometry types, " << n_components_ << " components");

    build_cell_map();
}

const FiniteElement* FESpace::get_fe(Index global_cell_idx) const {
    if (global_cell_idx < 0 ||
        static_cast<SizeType>(global_cell_idx) >= cell_locations_.size()) {
        return nullptr;
    }

    GeometryType geom_type = cell_locations_[global_cell_idx].geom_type;
    auto it = fe_by_geom_.find(geom_type);
    return it != fe_by_geom_.end() ? it->second.get() : nullptr;
}

int FESpace::dofs_per_cell(GeometryType geom_type) const {
    auto it = fe_by_geom_.find(geom_type);
    return it != fe_by_geom_.end() ? it->second->dofs_per_cell() : 0;
}

void FESpace::get_cell_dofs(SizeType block_idx, SizeType local_elem_idx,
                            std::vector<Index>& dofs) const {
    dofs.clear();

    if (block_idx >= mesh_->num_cell_blocks()) {
        return;
    }

    const auto& blocks = mesh_->cell_blocks();
    const auto& block = blocks[block_idx];

    GeometryType geom_type = to_geometry_type(block.type());
    auto fe_it = fe_by_geom_.find(geom_type);
    if (fe_it == fe_by_geom_.end()) {
        return;
    }

    const FiniteElement* fe = fe_it->second.get();
    auto vertices = block.element_vertices(local_elem_idx);

    int n_nodes = fe->dofs_per_cell() / n_components_;
    dofs.resize(fe->dofs_per_cell());

    // For linear H1 elements: dofs = vertices * components
    for (int i = 0; i < n_nodes && i < static_cast<int>(vertices.size()); ++i) {
        Index vertex_id = vertices[i];
        for (int c = 0; c < n_components_; ++c) {
            dofs[i * n_components_ + c] = vertex_id * n_components_ + c;
        }
    }

    // Handle higher-order elements (additional nodes)
    if (n_nodes > static_cast<int>(vertices.size())) {
        Index next_node = static_cast<Index>(mesh_->num_vertices());
        for (int i = static_cast<int>(vertices.size()); i < n_nodes; ++i, ++next_node) {
            for (int c = 0; c < n_components_; ++c) {
                dofs[i * n_components_ + c] = next_node * n_components_ + c;
            }
        }
    }
}

void FESpace::initialize() {
    // For H1 nodal elements: dofs are on nodes (vertices for linear)
    Index n_nodes = static_cast<Index>(mesh_->num_vertices());
    n_dofs_ = n_nodes * n_components_;

    MPFEM_INFO("FESpace initialized: " << n_dofs_ << " dofs ("
               << n_nodes << " nodes x " << n_components_ << " components)");
}

void FESpace::build_cell_map() {
    SizeType n_cells = mesh_->num_cells();
    cell_locations_.resize(n_cells);

    Index global_idx = 0;
    const auto& blocks = mesh_->cell_blocks();

    for (SizeType b = 0; b < blocks.size(); ++b) {
        const auto& block = blocks[b];
        GeometryType geom_type = to_geometry_type(block.type());

        for (SizeType i = 0; i < block.size(); ++i, ++global_idx) {
            cell_locations_[global_idx].block_idx = b;
            cell_locations_[global_idx].local_idx = i;
            cell_locations_[global_idx].geom_type = geom_type;
        }
    }

    MPFEM_INFO("Built cell map: " << cell_locations_.size() << " cells");
}

} // namespace mpfem
/**
 * @file field_space.cpp
 * @brief Field space implementation
 */

#include "field_space.hpp"
#include "mesh/element.hpp"
#include "core/logger.hpp"

namespace mpfem {

FieldSpace::FieldSpace(const FieldID& name, const Mesh* mesh, 
                       int n_components, int order)
    : name_(name)
    , mesh_(mesh)
    , n_components_(n_components)
    , order_(order)
    , n_dofs_(0)
{
    if (!mesh_) {
        MPFEM_ERROR("FieldSpace requires non-null mesh");
        return;
    }
    
    // For Lagrange elements on vertices: n_dofs = n_nodes * n_components
    n_dofs_ = static_cast<Index>(mesh_->num_vertices()) * n_components_;
    
    // Initialize solution vector
    solution_.setZero(n_dofs_);
    
    // Build cell DoF table
    build_cell_dof_table();
    
    MPFEM_INFO("Created field '" << name_ << "' with " << n_dofs_ 
               << " DoFs (" << n_components_ << " components, order " << order_ << ")");
}

void FieldSpace::get_cell_dofs(Index cell_id, std::vector<Index>& dofs) const {
    dofs.clear();
    
    if (cell_id < 0 || cell_id >= static_cast<Index>(cell_dof_table_.n_cells())) {
        return;
    }
    
    // Get vertex DoFs for this cell
    SizeType n = cell_dof_table_.dofs_per_cell(static_cast<SizeType>(cell_id));
    dofs.reserve(n);
    
    for (SizeType i = 0; i < n; ++i) {
        dofs.push_back(cell_dof_table_(static_cast<SizeType>(cell_id), static_cast<int>(i)));
    }
}

int FieldSpace::dofs_per_cell() const {
    // For Lagrange elements: number of vertices * n_components
    // This is approximate - actual value depends on cell type
    if (mesh_->cell_blocks().empty()) return 0;
    
    // Get average vertices per cell (rough estimate)
    SizeType total_cells = mesh_->num_cells();
    if (total_cells == 0) return 0;
    
    // Count vertices for first cell type
    for (const auto& block : mesh_->cell_blocks()) {
        if (block.size() > 0) {
            int verts_per_cell = block.nodes_per_element();
            return verts_per_cell * n_components_;
        }
    }
    return 0;
}

void FieldSpace::add_dirichlet_bc(Index boundary_id, Scalar value, int component) {
    // Find all nodes on this boundary
    for (const auto& block : mesh_->face_blocks()) {
        for (SizeType e = 0; e < block.size(); ++e) {
            if (block.entity_id(e) == boundary_id) {
                auto verts = block.element_vertices(e);
                for (Index v : verts) {
                    if (component < 0) {
                        // All components
                        for (int c = 0; c < n_components_; ++c) {
                            Index dof = node_dof(v, c);
                            if (constrained_dofs_.insert(dof).second) {
                                dirichlet_bcs_.emplace_back(dof, value);
                            }
                        }
                    } else if (component < n_components_) {
                        Index dof = node_dof(v, component);
                        if (constrained_dofs_.insert(dof).second) {
                            dirichlet_bcs_.emplace_back(dof, value);
                        }
                    }
                }
            }
        }
    }
}

void FieldSpace::apply_bcs_to_system(SparseMatrix& K, DynamicVector& f) const {
    // Apply Dirichlet BCs by modifying matrix diagonal and RHS
    // This is a simplified approach - for proper implementation,
    // we would zero out rows and set diagonal to 1
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    // Copy existing matrix
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            // Skip rows for constrained DoFs (will be overwritten)
            if (!is_constrained(it.row())) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    
    // Add diagonal entries for constrained DoFs
    for (const auto& [dof, value] : dirichlet_bcs_) {
        triplets.emplace_back(dof, dof, 1.0);
        f[dof] = value;
    }
    
    // Rebuild matrix
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
}

void FieldSpace::build_cell_dof_table() {
    SizeType n_cells = mesh_->num_cells();
    if (n_cells == 0) return;
    
    // Build DoF table for each cell
    std::vector<int> dofs_per_cell_vec;
    dofs_per_cell_vec.reserve(n_cells);
    
    for (const auto& block : mesh_->cell_blocks()) {
        int dim = element_dimension(block.type());
        if (dim < 3) continue;
        
        int verts_per_cell = block.nodes_per_element();
        int dofs = verts_per_cell * n_components_;
        
        for (SizeType i = 0; i < block.size(); ++i) {
            dofs_per_cell_vec.push_back(dofs);
        }
    }
    
    cell_dof_table_ = DoFTable(dofs_per_cell_vec);
    
    // Fill DoF indices
    SizeType cell_idx = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        int dim = element_dimension(block.type());
        if (dim < 3) continue;
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_idx) {
            auto verts = block.element_vertices(e);
            std::vector<Index> cell_dofs;
            cell_dofs.reserve(verts.size() * n_components_);
            
            for (Index v : verts) {
                for (int c = 0; c < n_components_; ++c) {
                    cell_dofs.push_back(node_dof(v, c));
                }
            }
            
            cell_dof_table_.set_cell_dofs(cell_idx, cell_dofs);
        }
    }
    
    MPFEM_INFO("Built cell DoF table for field '" << name_ 
               << "': " << cell_dof_table_.n_cells() << " cells");
}

} // namespace mpfem

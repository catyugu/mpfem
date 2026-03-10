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
    
    n_dofs_ = static_cast<Index>(mesh_->num_vertices()) * n_components_;
    solution_.setZero(n_dofs_);
    build_cell_dof_table();
    
    MPFEM_INFO("Created field '" << name_ << "' with " << n_dofs_ 
               << " DoFs (" << n_components_ << " components, order " << order_ << ")");
}

// ============================================================
// Solution Value Access
// ============================================================

Scalar FieldSpace::value_at_node(Index node_id) const {
    if (n_components_ != 1) return 0.0;
    Index dof = node_dof(node_id, 0);
    return (dof >= 0 && dof < solution_.size()) ? solution_[dof] : 0.0;
}

void FieldSpace::set_value_at_node(Index node_id, Scalar value) {
    if (n_components_ != 1) return;
    Index dof = node_dof(node_id, 0);
    if (dof >= 0 && dof < solution_.size()) {
        solution_[dof] = value;
    }
}

Tensor<1, 3> FieldSpace::vector_at_node(Index node_id) const {
    Tensor<1, 3> result = Tensor<1, 3>::Zero();
    for (int c = 0; c < n_components_ && c < 3; ++c) {
        Index dof = node_dof(node_id, c);
        if (dof >= 0 && dof < solution_.size()) {
            result[c] = solution_[dof];
        }
    }
    return result;
}

void FieldSpace::set_vector_at_node(Index node_id, const Tensor<1, 3>& value) {
    for (int c = 0; c < n_components_ && c < 3; ++c) {
        Index dof = node_dof(node_id, c);
        if (dof >= 0 && dof < solution_.size()) {
            solution_[dof] = value[c];
        }
    }
}

Scalar FieldSpace::component_at_node(Index node_id, int component) const {
    if (component < 0 || component >= n_components_) return 0.0;
    Index dof = node_dof(node_id, component);
    return (dof >= 0 && dof < solution_.size()) ? solution_[dof] : 0.0;
}

void FieldSpace::set_component_at_node(Index node_id, int component, Scalar value) {
    if (component < 0 || component >= n_components_) return;
    Index dof = node_dof(node_id, component);
    if (dof >= 0 && dof < solution_.size()) {
        solution_[dof] = value;
    }
}

// ============================================================
// Boundary Conditions
// ============================================================

bool FieldSpace::is_node_constrained(Index node_id) const {
    for (int c = 0; c < n_components_; ++c) {
        if (is_constrained(node_dof(node_id, c))) return true;
    }
    return false;
}

Index FieldSpace::n_constrained_nodes() const {
    std::unordered_set<Index> nodes;
    for (const auto& [dof, value] : dirichlet_bcs_) {
        nodes.insert(dof / n_components_);
    }
    return static_cast<Index>(nodes.size());
}

void FieldSpace::add_dirichlet_bc(Index boundary_id, Scalar value, int component) {
    for (const auto& block : mesh_->face_blocks()) {
        for (SizeType e = 0; e < block.size(); ++e) {
            if (block.entity_id(e) == boundary_id) {
                auto verts = block.element_vertices(e);
                for (Index v : verts) {
                    if (component < 0) {
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
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            if (!is_constrained(it.row())) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    
    for (const auto& [dof, value] : dirichlet_bcs_) {
        triplets.emplace_back(dof, dof, 1.0);
        f[dof] = value;
    }
    
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
}

// ============================================================
// DoF Index Access
// ============================================================

void FieldSpace::get_cell_dofs(Index cell_id, std::vector<Index>& dofs) const {
    dofs.clear();
    
    if (cell_id < 0 || cell_id >= static_cast<Index>(cell_dof_table_.n_cells())) {
        return;
    }
    
    SizeType n = cell_dof_table_.dofs_per_cell(static_cast<SizeType>(cell_id));
    dofs.reserve(n);
    
    for (SizeType i = 0; i < n; ++i) {
        dofs.push_back(cell_dof_table_(static_cast<SizeType>(cell_id), static_cast<int>(i)));
    }
}

int FieldSpace::dofs_per_cell() const {
    if (mesh_->cell_blocks().empty()) return 0;
    
    for (const auto& block : mesh_->cell_blocks()) {
        if (block.size() > 0) {
            int verts_per_cell = block.nodes_per_element();
            return verts_per_cell * n_components_;
        }
    }
    return 0;
}

// ============================================================
// Private Implementation
// ============================================================

void FieldSpace::build_cell_dof_table() {
    SizeType n_cells = mesh_->num_cells();
    if (n_cells == 0) return;
    
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

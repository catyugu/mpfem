/**
 * @file electrostatics.cpp
 * @brief Electrostatics physics assembly implementation
 */

#include "electrostatics.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_base.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void ElectrostaticsAssembly::initialize(const Mesh* mesh,
                                        const DoFHandler* dof_handler,
                                        const MaterialDB* mat_db,
                                        const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, dof_handler, mat_db);
    
    // Copy boundary conditions
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Electrostatics initialized with " << bcs_.size() << " boundary conditions");
}

void ElectrostaticsAssembly::assemble_stiffness(SparseMatrix& K) {
    const FESpace* fe_space = dof_handler_->fe_space();
    Index n_dofs = dof_handler_->n_dofs();
    
    // Use triplet list for assembly
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(n_dofs * 20);
    
    // Create FEValues for gradient computation
    UpdateFlags flags = UpdateFlags::UpdateDefault;
    FEValues fe_values(nullptr, flags);
    
    std::vector<Index> cell_dofs;
    DynamicMatrix K_local;
    
    // Iterate over all cell blocks
    SizeType global_cell_idx = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        int dim = element_dimension(block.type());
        if (dim < 3) continue;  // Skip non-3D elements
        
        GeometryType geom_type = to_geometry_type(block.type());
        const FiniteElement* fe = fe_space->get_fe(static_cast<Index>(global_cell_idx));
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        fe_values = FEValues(fe, flags);
        
        for (SizeType e = 0; e < block.size(); ++e, ++global_cell_idx) {
            // Get cell DoFs
            dof_handler_->get_cell_dofs(static_cast<Index>(global_cell_idx), cell_dofs);
            if (cell_dofs.empty()) continue;
            
            // Initialize FEValues for this cell
            fe_values.reinit(*mesh_, static_cast<Index>(global_cell_idx));
            
            // Get domain ID and material
            Index domain_id = block.entity_id(e);
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            // Get conductivity
            MaterialEvaluator evaluator;
            if (temperature_field_) {
                // Get average temperature for this cell
                Scalar avg_temp = 0.0;
                for (Index dof : cell_dofs) {
                    if (dof < temperature_field_->size()) {
                        avg_temp += (*temperature_field_)[dof];
                    }
                }
                avg_temp /= cell_dofs.size();
                evaluator.set_temperature(avg_temp);
            }
            Tensor<2, 3> sigma = material->get_conductivity(evaluator);
            
            // Assemble local stiffness matrix
            int n = fe->dofs_per_cell();
            K_local.setZero(n, n);
            
            for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
                Scalar jxw = fe_values.JxW(q);
                
                for (int i = 0; i < n; ++i) {
                    const auto& grad_i = fe_values.shape_grad(i, q);
                    for (int j = 0; j < n; ++j) {
                        const auto& grad_j = fe_values.shape_grad(j, q);
                        
                        // K_ij = ∫ (σ ∇N_i · ∇N_j) dx
                        Scalar val = 0.0;
                        for (int d1 = 0; d1 < 3; ++d1) {
                            for (int d2 = 0; d2 < 3; ++d2) {
                                val += sigma(d1, d2) * grad_i[d1] * grad_j[d2];
                            }
                        }
                        K_local(i, j) += val * jxw;
                    }
                }
            }
            
            // Add to global matrix
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    triplets.emplace_back(cell_dofs[i], cell_dofs[j], K_local(i, j));
                }
            }
        }
    }
    
    // Build sparse matrix
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Electrostatics stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void ElectrostaticsAssembly::assemble_rhs(DynamicVector& f) {
    Index n_dofs = dof_handler_->n_dofs();
    f.setZero(n_dofs);
    
    // No source terms for steady-state electrostatics
    // RHS contributions come from Neumann BCs
}

void ElectrostaticsAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    // Apply Dirichlet boundary conditions
    // For each BC, set the diagonal to 1 and RHS to the prescribed value
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    // Copy existing matrix
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    // Track constrained DoFs
    std::unordered_set<Index> constrained_dofs;
    
    for (const auto& bc : bcs_) {
        if (bc.kind == "voltage") {
            // Get voltage value from params
            Scalar value = 0.0;
            auto it = bc.params.find("value");
            if (it != bc.params.end()) {
                // Parse the value (could be a variable name or number)
                try {
                    value = std::stod(it->second);
                } catch (...) {
                    // Keep default 0.0
                }
            }
            
            // Get boundary vertices
            for (Index bnd_id : bc.ids) {
                // Find faces with this boundary ID
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            // Get vertices of this face
                            auto verts = block.element_vertices(e);
                            for (Index v : verts) {
                                // For scalar field, DoF = vertex ID
                                Index dof = v;
                                if (constrained_dofs.insert(dof).second) {
                                    // Set diagonal to 1
                                    triplets.emplace_back(dof, dof, 1.0);
                                    // Set RHS to prescribed value
                                    f[dof] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Rebuild matrix with modifications
    Index n_dofs = dof_handler_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for electrostatics");
}

DynamicVector ElectrostaticsAssembly::get_field_gradient(const DynamicVector& solution) const {
    // Compute electric field E = -∇V at each node
    // This is a simplified version - proper implementation would
    // project from quadrature points to nodes
    
    DynamicVector grad;
    Index n_nodes = mesh_->num_vertices();
    grad.resize(n_nodes * 3);
    grad.setZero();
    
    // TODO: Implement proper gradient projection
    // For now, return zeros
    
    return grad;
}

} // namespace mpfem

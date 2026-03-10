/**
 * @file heat_transfer.cpp
 * @brief Heat transfer physics assembly implementation
 */

#include "heat_transfer.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_base.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void HeatTransferAssembly::initialize(const Mesh* mesh,
                                      const DoFHandler* dof_handler,
                                      const MaterialDB* mat_db,
                                      const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, dof_handler, mat_db);
    
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Heat transfer initialized with " << bcs_.size() << " boundary conditions");
}

void HeatTransferAssembly::assemble_stiffness(SparseMatrix& K) {
    const FESpace* fe_space = dof_handler_->fe_space();
    Index n_dofs = dof_handler_->n_dofs();
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(n_dofs * 20);
    
    UpdateFlags flags = UpdateFlags::UpdateDefault;
    FEValues fe_values(nullptr, flags);
    
    std::vector<Index> cell_dofs;
    DynamicMatrix K_local;
    
    SizeType global_cell_idx = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        int dim = element_dimension(block.type());
        if (dim < 3) continue;
        
        GeometryType geom_type = to_geometry_type(block.type());
        const FiniteElement* fe = fe_space->get_fe(static_cast<Index>(global_cell_idx));
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        fe_values = FEValues(fe, flags);
        
        for (SizeType e = 0; e < block.size(); ++e, ++global_cell_idx) {
            dof_handler_->get_cell_dofs(static_cast<Index>(global_cell_idx), cell_dofs);
            if (cell_dofs.empty()) continue;
            
            fe_values.reinit(*mesh_, static_cast<Index>(global_cell_idx));
            
            Index domain_id = block.entity_id(e);
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            MaterialEvaluator evaluator;
            Tensor<2, 3> k = material->get_thermal_conductivity(evaluator);
            
            int n = fe->dofs_per_cell();
            K_local.setZero(n, n);
            
            for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
                Scalar jxw = fe_values.JxW(q);
                
                for (int i = 0; i < n; ++i) {
                    const auto& grad_i = fe_values.shape_grad(i, q);
                    for (int j = 0; j < n; ++j) {
                        const auto& grad_j = fe_values.shape_grad(j, q);
                        
                        Scalar val = 0.0;
                        for (int d1 = 0; d1 < 3; ++d1) {
                            for (int d2 = 0; d2 < 3; ++d2) {
                                val += k(d1, d2) * grad_i[d1] * grad_j[d2];
                            }
                        }
                        K_local(i, j) += val * jxw;
                    }
                }
            }
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    triplets.emplace_back(cell_dofs[i], cell_dofs[j], K_local(i, j));
                }
            }
        }
    }
    
    // Add convection boundary contributions to stiffness matrix
    for (const auto& bc : bcs_) {
        if (bc.kind == "convection") {
            Scalar h = 0.0;
            auto it = bc.params.find("h");
            if (it != bc.params.end()) {
                try { h = std::stod(it->second); } catch (...) {}
            }
            
            for (Index bnd_id : bc.ids) {
                // Find faces with this boundary ID and add convection term
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            // Add h * ∫ N_i * N_j dS to the stiffness matrix
                            // This is a simplified implementation
                            auto verts = block.element_vertices(e);
                            int n = static_cast<int>(verts.size());
                            
                            // Simple lumped mass matrix for boundary
                            Scalar area = 1.0; // TODO: compute actual face area
                            for (int i = 0; i < n; ++i) {
                                for (int j = 0; j < n; ++j) {
                                    // Diagonal approximation
                                    if (i == j) {
                                        Index dof_i = verts[i];
                                        triplets.emplace_back(dof_i, dof_i, h * area / n);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Heat transfer stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void HeatTransferAssembly::assemble_rhs(DynamicVector& f) {
    Index n_dofs = dof_handler_->n_dofs();
    f.setZero(n_dofs);
    
    // Add volumetric heat source
    if (heat_source_) {
        // Heat source should be defined per node
        for (Index i = 0; i < std::min(n_dofs, static_cast<Index>(heat_source_->size())); ++i) {
            f[i] += (*heat_source_)[i];
        }
    }
    
    // Add convection boundary contributions to RHS
    for (const auto& bc : bcs_) {
        if (bc.kind == "convection") {
            Scalar h = 0.0;
            Scalar T_inf = 293.15;
            
            auto hit = bc.params.find("h");
            if (hit != bc.params.end()) {
                try { h = std::stod(hit->second); } catch (...) {}
            }
            auto tit = bc.params.find("T_inf");
            if (tit != bc.params.end()) {
                try { T_inf = std::stod(tit->second); } catch (...) {}
            }
            
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            int n = static_cast<int>(verts.size());
                            
                            Scalar area = 1.0; // TODO: compute actual face area
                            for (int i = 0; i < n; ++i) {
                                Index dof = verts[i];
                                if (dof < n_dofs) {
                                    f[dof] += h * T_inf * area / n;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void HeatTransferAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    // Apply thermal insulation (natural BC - no action needed)
    // Apply fixed temperature (Dirichlet BC)
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    std::unordered_set<Index> constrained_dofs;
    
    for (const auto& bc : bcs_) {
        if (bc.kind == "fixed_temperature" || bc.kind == "temperature") {
            Scalar T = 293.15;
            auto it = bc.params.find("value");
            if (it != bc.params.end()) {
                try { T = std::stod(it->second); } catch (...) {}
            }
            
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            for (Index v : verts) {
                                Index dof = v;
                                if (constrained_dofs.insert(dof).second) {
                                    triplets.emplace_back(dof, dof, 1.0);
                                    f[dof] = T;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Index n_dofs = dof_handler_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for heat transfer");
}

} // namespace mpfem

/**
 * @file heat_transfer.cpp
 * @brief Heat transfer physics assembly implementation
 */

#include "heat_transfer.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_cache.hpp"
#include "fem/fe_collection.hpp"
#include "material/material_database.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void HeatTransferAssembly::initialize(const Mesh* mesh,
                                      const FieldSpace* field,
                                      const MaterialDB* mat_db,
                                      const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, field, mat_db);
    
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Heat transfer initialized with " << bcs_.size() << " boundary conditions");
}

void HeatTransferAssembly::assemble_stiffness(SparseMatrix& K) {
    if (!field_ || !mesh_) return;
    
    Index n_dofs = field_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setZero();
    
    // Use FECache to avoid repeated FE creation
    auto& fe_cache = FECache::instance();
    const int order = field_->order();
    const int n_comp = field_->n_components();
    
    // Pre-get FE for each geometry type
    std::unordered_map<GeometryType, std::shared_ptr<const FiniteElement>> fe_map;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        if (fe_map.find(geom_type) == fe_map.end()) {
            fe_map[geom_type] = fe_cache.get(geom_type, order, n_comp);
        }
    }
    
    DynamicMatrix K_local;
    std::vector<Eigen::Triplet<Scalar>> triplets;
    std::vector<Index> cell_dofs;
    cell_dofs.reserve(27);
    
    Index cell_id = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        auto fe = fe_map[geom_type];
        if (!fe) {
            cell_id += static_cast<Index>(block.size());
            continue;
        }
        
        FEValues fe_values(fe.get(), UpdateFlags::UpdateDefault);
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_id) {
            Index domain_id = mesh_->get_cell_domain_id(cell_id);
            
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            MaterialEvaluator evaluator;
            Tensor<2, 3> k = material->get_thermal_conductivity(evaluator);
            
            fe_values.reinit(*field_, cell_id);
            
            int n = fe_values.dofs_per_cell();
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
            
            cell_dofs.clear();
            field_->get_cell_dofs(cell_id, cell_dofs);
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (std::abs(K_local(i, j)) > 1e-15) {
                        triplets.emplace_back(cell_dofs[i], cell_dofs[j], K_local(i, j));
                    }
                }
            }
        }
    }
    
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Heat transfer stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void HeatTransferAssembly::assemble_rhs(DynamicVector& f) {
    Index n_dofs = field_->n_dofs();
    f.setZero(n_dofs);
    
    // Add heat source (e.g., from Joule heating)
    if (heat_source_) {
        for (Index i = 0; i < std::min(n_dofs, static_cast<Index>(heat_source_->size())); ++i) {
            f[i] += (*heat_source_)[i];
        }
    }
    
    // Note: Convection BC contribution is handled in apply_boundary_conditions
    // to properly combine matrix and RHS contributions
}

void HeatTransferAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    std::unordered_set<Index> constrained_dofs;
    Index n_dofs = field_->n_dofs();
    
    // Process Dirichlet BCs (fixed temperature)
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
                                if (constrained_dofs.insert(v).second) {
                                    triplets.emplace_back(v, v, 1.0);
                                    f[v] = T;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Process convection BCs (Robin BC): -k ∂T/∂n = h(T - T_inf)
    // Contribution to matrix: ∫_Γ h N_i N_j dS
    // Contribution to RHS: ∫_Γ h T_inf N_i dS
    // Using lumped mass approximation for surface integral
    for (const auto& bc : bcs_) {
        if (bc.kind == "convection") {
            Scalar h = 0.0, T_inf = 293.15;
            
            auto hit = bc.params.find("h");
            if (hit != bc.params.end()) {
                try { h = std::stod(hit->second); } catch (...) {}
            }
            auto tit = bc.params.find("T_inf");
            if (tit != bc.params.end()) {
                try { T_inf = std::stod(tit->second); } catch (...) {}
            }
            
            if (h <= 0) continue;
            
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) != bnd_id) continue;
                        
                        // Get face vertices
                        auto verts = block.element_vertices(e);
                        int n = static_cast<int>(verts.size());
                        
                        // Compute face area
                        Scalar face_area = 0.0;
                        if (n == 3) {  // Triangle
                            auto p0 = mesh_->vertex(verts[0]);
                            auto p1 = mesh_->vertex(verts[1]);
                            auto p2 = mesh_->vertex(verts[2]);
                            // Cross product for 3D triangle area
                            auto v1 = p1 - p0;
                            auto v2 = p2 - p0;
                            auto cross = Tensor<1, 3>(
                                v1.y() * v2.z() - v1.z() * v2.y(),
                                v1.z() * v2.x() - v1.x() * v2.z(),
                                v1.x() * v2.y() - v1.y() * v2.x()
                            );
                            face_area = 0.5 * cross.norm();
                        } else if (n == 4) {  // Quadrilateral
                            // Split into two triangles
                            auto p0 = mesh_->vertex(verts[0]);
                            auto p1 = mesh_->vertex(verts[1]);
                            auto p2 = mesh_->vertex(verts[2]);
                            auto p3 = mesh_->vertex(verts[3]);
                            
                            Tensor<1, 3> v1 = p1 - p0;
                            Tensor<1, 3> v2 = p3 - p0;
                            Tensor<1, 3> cross1(
                                v1.y() * v2.z() - v1.z() * v2.y(),
                                v1.z() * v2.x() - v1.x() * v2.z(),
                                v1.x() * v2.y() - v1.y() * v2.x()
                            );
                            
                            v1 = p2 - p1;
                            v2 = p0 - p1;
                            Tensor<1, 3> cross2(
                                v1.y() * v2.z() - v1.z() * v2.y(),
                                v1.z() * v2.x() - v1.x() * v2.z(),
                                v1.x() * v2.y() - v1.y() * v2.x()
                            );
                            
                            face_area = 0.5 * (cross1.norm() + cross2.norm());
                        }
                        
                        // Lumped mass approximation: ∫_face N_i dS ≈ face_area/n
                        // Matrix contribution: h * face_area/n for each node
                        // RHS contribution: h * T_inf * face_area/n
                        Scalar contrib = h * face_area / n;
                        
                        for (int i = 0; i < n; ++i) {
                            Index dof_i = verts[i];
                            if (dof_i >= n_dofs || constrained_dofs.count(dof_i) > 0) continue;
                            
                            // Add to matrix diagonal (h * ∫ N_i^2 dS ≈ h * face_area/n)
                            triplets.emplace_back(dof_i, dof_i, contrib);
                            
                            // Add to RHS: h * T_inf * ∫ N_i dS
                            f[dof_i] += contrib * T_inf;
                        }
                    }
                }
            }
        }
    }
    
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for heat transfer");
}

} // namespace mpfem

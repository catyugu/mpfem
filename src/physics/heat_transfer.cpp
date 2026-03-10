/**
 * @file heat_transfer.cpp
 * @brief Heat transfer physics assembly implementation
 */

#include "heat_transfer.hpp"
#include "assembly/fe_values.hpp"
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
    
    auto fe = create_fe(GeometryType::Tetrahedron, field_->order(), field_->n_components());
    if (!fe) return;
    
    FEValues fe_values(fe.get(), UpdateFlags::UpdateDefault);
    DynamicMatrix K_local;
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    
    for (Index cell_id = 0; cell_id < static_cast<Index>(mesh_->num_cells()); ++cell_id) {
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
        
        std::vector<Index> cell_dofs;
        field_->get_cell_dofs(cell_id, cell_dofs);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::abs(K_local(i, j)) > 1e-15) {
                    triplets.emplace_back(cell_dofs[i], cell_dofs[j], K_local(i, j));
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
    
    if (heat_source_) {
        for (Index i = 0; i < std::min(n_dofs, static_cast<Index>(heat_source_->size())); ++i) {
            f[i] += (*heat_source_)[i];
        }
    }
    
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
            
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            int n = static_cast<int>(verts.size());
                            
                            for (int i = 0; i < n; ++i) {
                                Index dof = verts[i];
                                if (dof < n_dofs) {
                                    f[dof] += h * T_inf / n;  // Simplified
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
    
    Index n_dofs = field_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for heat transfer");
}

} // namespace mpfem

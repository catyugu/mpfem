/**
 * @file electrostatics.cpp
 * @brief Electrostatics physics assembly implementation
 */

#include "electrostatics.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_cache.hpp"
#include "fem/fe_collection.hpp"
#include "material/material_database.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void ElectrostaticsAssembly::initialize(const Mesh* mesh,
                                        const FieldSpace* field,
                                        const MaterialDB* mat_db,
                                        const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, field, mat_db);
    
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Electrostatics initialized with " << bcs_.size() << " boundary conditions");
}

void ElectrostaticsAssembly::assemble_stiffness(SparseMatrix& K) {
    if (!field_ || !mesh_) {
        MPFEM_ERROR("Electrostatics: field not initialized");
        return;
    }
    
    Index n_dofs = field_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setZero();
    
    // Use FECache to avoid repeated FE creation
    auto& fe_cache = FECache::instance();
    const int order = field_->order();
    const int n_comp = field_->n_components();
    
    DynamicMatrix K_local;
    std::vector<Eigen::Triplet<Scalar>> triplets;
    std::vector<Index> cell_dofs;
    cell_dofs.reserve(27);  // Max for Tet10
    
    // Pre-get FE for each geometry type
    std::unordered_map<GeometryType, std::shared_ptr<const FiniteElement>> fe_map;
    
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        if (fe_map.find(geom_type) == fe_map.end()) {
            fe_map[geom_type] = fe_cache.get(geom_type, order, n_comp);
        }
    }
    
    Index cell_id = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        auto fe = fe_map[geom_type];
        if (!fe) {
            cell_id += static_cast<Index>(block.size());
            continue;
        }
        
        FEValues fe_values(fe.get(), UpdateFlags::UpdateDefault);
        if (field_registry_) {
            fe_values.set_field_registry(field_registry_);
        }
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_id) {
            Index domain_id = mesh_->get_cell_domain_id(cell_id);
            
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            fe_values.reinit(*field_, cell_id);
            
            int n = fe_values.dofs_per_cell();
            K_local.setZero(n, n);
            
            // 逐积分点计算
            for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
                Scalar jxw = fe_values.JxW(q);
                
                // 在该积分点获取温度（如果温度场存在）
                MaterialEvaluator evaluator;
                if (field_registry_) {
                    Scalar T = fe_values.field_value("temperature", q);
                    evaluator.set_temperature(T);
                }
                
                // 基于该积分点的温度计算电导率
                Tensor<2, 3> sigma = material->get_conductivity(evaluator);
                
                for (int i = 0; i < n; ++i) {
                    const auto& grad_i = fe_values.shape_grad(i, q);
                    for (int j = 0; j < n; ++j) {
                        const auto& grad_j = fe_values.shape_grad(j, q);
                        
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
    
    MPFEM_INFO("Electrostatics stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void ElectrostaticsAssembly::assemble_rhs(DynamicVector& f) {
    f.setZero(field_->n_dofs());
}

void ElectrostaticsAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    std::unordered_set<Index> constrained_dofs;
    
    for (const auto& bc : bcs_) {
        if (bc.kind == "voltage") {
            Scalar value = 0.0;
            auto it = bc.params.find("value");
            if (it != bc.params.end()) {
                try { value = std::stod(it->second); } catch (...) {}
            }
            
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            for (Index v : verts) {
                                if (constrained_dofs.insert(v).second) {
                                    triplets.emplace_back(v, v, 1.0);
                                    f[v] = value;
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
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for electrostatics");
}

std::vector<Scalar> ElectrostaticsAssembly::compute_joule_heating(const FieldRegistry& registry) const {
    Index n_nodes = mesh_->num_vertices();
    std::vector<Scalar> Q(n_nodes, 0.0);
    
    const FieldSpace* V_field = registry.get_field("electric_potential");
    if (!V_field) {
        MPFEM_WARN("Electric potential field not found for Joule heating computation");
        return Q;
    }
    
    // Use FECache for better performance
    auto& fe_cache = FECache::instance();
    
    // Pre-get FE for each geometry type
    std::unordered_map<GeometryType, std::shared_ptr<const FiniteElement>> fe_map;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        if (fe_map.find(geom_type) == fe_map.end()) {
            fe_map[geom_type] = fe_cache.get(geom_type, V_field->order(), V_field->n_components());
        }
    }
    
    std::vector<Scalar> node_weights(n_nodes, 0.0);
    std::vector<Index> vertices;
    vertices.reserve(10);  // Max for Tet10
    
    Index cell_id = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        auto fe = fe_map[geom_type];
        if (!fe) {
            cell_id += static_cast<Index>(block.size());
            continue;
        }
        
        FEValues fe_values(fe.get(), UpdateFlags::UpdateDefault);
        fe_values.set_field_registry(&registry);
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_id) {
            Index domain_id = mesh_->get_cell_domain_id(cell_id);
            
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            fe_values.reinit(*V_field, cell_id);
            
            vertices.clear();
            vertices = mesh_->get_cell_vertices(cell_id);
            
            for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
                Scalar jxw = fe_values.JxW(q);
                
                // 获取该积分点的温度
                MaterialEvaluator evaluator;
                Scalar T = fe_values.field_value("temperature", q);
                evaluator.set_temperature(T);
                
                // 获取电导率
                Tensor<2, 3> sigma = material->get_conductivity(evaluator);
                
                // 计算电场强度 E = -∇V
                Tensor<1, 3> E = fe_values.gradient(q);
                
                // 焦耳热 Q = σ|E|²
                Scalar q_joule = 0.0;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        q_joule += sigma(i, j) * E[i] * E[j];
                    }
                }
                
                // 分配到节点
                for (size_t i = 0; i < vertices.size(); ++i) {
                    Index node = vertices[i];
                    Scalar N_i = fe_values.shape_value(static_cast<int>(i), q);
                    Q[node] += q_joule * N_i * jxw;
                    node_weights[node] += N_i * jxw;
                }
            }
        }
    }
    
    // 归一化
    for (Index n = 0; n < n_nodes; ++n) {
        if (node_weights[n] > 1e-15) {
            Q[n] /= node_weights[n];
        }
    }
    
    return Q;
}

} // namespace mpfem
/**
 * @file solid_mechanics.cpp
 * @brief Solid mechanics physics assembly implementation
 */

#include "solid_mechanics.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_collection.hpp"
#include "material/material_database.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void SolidMechanicsAssembly::initialize(const Mesh* mesh,
                                        const FieldSpace* field,
                                        const MaterialDB* mat_db,
                                        const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, field, mat_db);
    
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Solid mechanics initialized with " << bcs_.size() << " boundary conditions");
}

void SolidMechanicsAssembly::assemble_stiffness(SparseMatrix& K) {
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
        
        Scalar E = material->get_youngs_modulus();
        Scalar nu = material->get_poissons_ratio();
        
        fe_values.reinit(*field_, cell_id);
        
        int n = fe_values.dofs_per_cell();
        int dim = 3;
        int n_nodes = n / dim;
        
        K_local.setZero(n, n);
        
        Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Scalar mu = E / (2.0 * (1.0 + nu));
        
        for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
            Scalar jxw = fe_values.JxW(q);
            
            for (int a = 0; a < n_nodes; ++a) {
                const auto& grad_a = fe_values.shape_grad(a, q);
                
                for (int b = 0; b < n_nodes; ++b) {
                    const auto& grad_b = fe_values.shape_grad(b, q);
                    
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            Scalar val = lambda * grad_a[i] * grad_b[j];
                            val += mu * (grad_a[j] * grad_b[i] + grad_a[i] * grad_b[j]);
                            K_local(a*dim + i, b*dim + j) += val * jxw;
                        }
                    }
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
    
    MPFEM_INFO("Solid mechanics stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void SolidMechanicsAssembly::assemble_rhs(DynamicVector& f) {
    Index n_dofs = field_->n_dofs();
    f.setZero(n_dofs);
    
    if (temperature_field_) {
        MPFEM_INFO("Thermal strain contribution not yet implemented");
    }
}

void SolidMechanicsAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    std::unordered_set<Index> constrained_dofs;
    int n_comp = field_->n_components();
    
    for (const auto& bc : bcs_) {
        if (bc.kind == "fixed_constraint") {
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            for (Index v : verts) {
                                for (int comp = 0; comp < n_comp; ++comp) {
                                    Index dof = v * n_comp + comp;
                                    if (constrained_dofs.insert(dof).second) {
                                        triplets.emplace_back(dof, dof, 1.0);
                                        f[dof] = 0.0;
                                    }
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
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for solid mechanics");
}

} // namespace mpfem

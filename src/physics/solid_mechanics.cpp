/**
 * @file solid_mechanics.cpp
 * @brief Solid mechanics physics assembly implementation
 */

#include "solid_mechanics.hpp"
#include "assembly/fe_values.hpp"
#include "fem/fe_base.hpp"
#include <Eigen/Sparse>

namespace mpfem {

void SolidMechanicsAssembly::initialize(const Mesh* mesh,
                                        const DoFHandler* dof_handler,
                                        const MaterialDB* mat_db,
                                        const PhysicsConfig& config) {
    PhysicsAssembly::initialize(mesh, dof_handler, mat_db);
    
    for (const auto& bc : config.boundaries) {
        bcs_.push_back(bc);
    }
    
    MPFEM_INFO("Solid mechanics initialized with " << bcs_.size() << " boundary conditions");
}

Tensor<4, 3> SolidMechanicsAssembly::compute_elasticity_tensor(Scalar E, Scalar nu) const {
    // Isotropic elasticity tensor in Voigt notation
    // D_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})
    // where λ = E ν / ((1+ν)(1-2ν)) and μ = E / (2(1+ν))
    
    Tensor<4, 3> D;
    Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Scalar mu = E / (2.0 * (1.0 + nu));
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    D(i, j, k, l) = lambda * (i == j ? 1.0 : 0.0) * (k == l ? 1.0 : 0.0)
                                  + mu * ((i == k ? 1.0 : 0.0) * (j == l ? 1.0 : 0.0)
                                        + (i == l ? 1.0 : 0.0) * (j == k ? 1.0 : 0.0));
                }
            }
        }
    }
    
    return D;
}

void SolidMechanicsAssembly::assemble_stiffness(SparseMatrix& K) {
    const FESpace* fe_space = dof_handler_->fe_space();
    Index n_dofs = dof_handler_->n_dofs();
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(n_dofs * 50);
    
    UpdateFlags flags = UpdateFlags::UpdateDefault;
    FEValues fe_values(nullptr, flags);
    
    std::vector<Index> cell_dofs;
    DynamicMatrix K_local;
    
    SizeType global_cell_idx = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        int dim = element_dimension(block.type());
        if (dim < 3) continue;
        
        const FiniteElement* fe = fe_space->get_fe(static_cast<Index>(global_cell_idx));
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        fe_values = FEValues(fe, flags);
        int n_comp = fe->n_components();
        int n = fe->dofs_per_cell();
        
        for (SizeType e = 0; e < block.size(); ++e, ++global_cell_idx) {
            dof_handler_->get_cell_dofs(static_cast<Index>(global_cell_idx), cell_dofs);
            if (cell_dofs.empty()) continue;
            
            fe_values.reinit(*mesh_, static_cast<Index>(global_cell_idx));
            
            Index domain_id = block.entity_id(e);
            auto mat_it = domain_material_map_.find(domain_id);
            if (mat_it == domain_material_map_.end()) continue;
            
            const Material* material = mat_db_->get(mat_it->second);
            if (!material) continue;
            
            Scalar E = material->get_youngs_modulus();
            Scalar nu = material->get_poissons_ratio();
            
            Tensor<4, 3> D = compute_elasticity_tensor(E, nu);
            
            K_local.setZero(n, n);
            
            for (int q = 0; q < fe_values.n_quadrature_points(); ++q) {
                Scalar jxw = fe_values.JxW(q);
                
                // For vector-valued FE, each DoF corresponds to a component
                // dof i * n_comp + comp -> shape function i, component comp
                for (int a = 0; a < n / n_comp; ++a) {
                    for (int i = 0; i < 3; ++i) {
                        int dof_a_i = a * n_comp + i;
                        const auto& grad_a = fe_values.shape_grad(a, q);
                        
                        for (int b = 0; b < n / n_comp; ++b) {
                            for (int j = 0; j < 3; ++j) {
                                int dof_b_j = b * n_comp + j;
                                const auto& grad_b = fe_values.shape_grad(b, q);
                                
                                // K_{ab}^{ij} = ∫ B_a^i : D : B_b^j dV
                                // where B_a^i is the strain-displacement vector for
                                // shape function a, component i
                                
                                Scalar val = 0.0;
                                for (int k = 0; k < 3; ++k) {
                                    for (int l = 0; l < 3; ++l) {
                                        // Strain from shape a, comp i
                                        // ε_{kl}^a = 0.5 * (∂N_a/∂x_l * δ_{ik} + ∂N_a/∂x_k * δ_{il})
                                        // But for stiffness we use: B_a^i_{kl} = ∂N_a/∂x_l * δ_{ik}
                                        
                                        for (int m = 0; m < 3; ++m) {
                                            for (int n = 0; n < 3; ++n) {
                                                // Simplified isotropic formulation
                                                val += D(k, l, m, n) * 
                                                       grad_a[l] * (i == k ? 0.5 : 0.0) *
                                                       grad_b[n] * (j == m ? 0.5 : 0.0);
                                            }
                                        }
                                    }
                                }
                                K_local(dof_a_i, dof_b_j) += val * jxw;
                            }
                        }
                    }
                }
            }
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (std::abs(K_local(i, j)) > 1e-15) {
                        triplets.emplace_back(cell_dofs[i], cell_dofs[j], K_local(i, j));
                    }
                }
            }
        }
    }
    
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Solid mechanics stiffness matrix assembled: " << K.nonZeros() << " non-zeros");
}

void SolidMechanicsAssembly::assemble_rhs(DynamicVector& f) {
    Index n_dofs = dof_handler_->n_dofs();
    f.setZero(n_dofs);
    
    // Add thermal strain contribution (if temperature field is set)
    if (temperature_field_) {
        // Thermal strain: ε_th = α (T - T_ref)
        // This contributes to the RHS as: f = ∫ B^T D ε_th dV
        
        // TODO: Implement thermal strain contribution
        MPFEM_INFO("Thermal strain contribution not yet implemented");
    }
}

void SolidMechanicsAssembly::apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) {
    // Apply fixed constraint (zero displacement) - Dirichlet BC
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(K.nonZeros());
    
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    
    std::unordered_set<Index> constrained_dofs;
    int n_comp = n_components();
    
    for (const auto& bc : bcs_) {
        if (bc.kind == "fixed_constraint") {
            for (Index bnd_id : bc.ids) {
                for (const auto& block : mesh_->face_blocks()) {
                    for (SizeType e = 0; e < block.size(); ++e) {
                        if (block.entity_id(e) == bnd_id) {
                            auto verts = block.element_vertices(e);
                            for (Index v : verts) {
                                // For vector field, DoFs are: v * n_comp + comp
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
    
    Index n_dofs = dof_handler_->n_dofs();
    K.resize(n_dofs, n_dofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    
    MPFEM_INFO("Applied " << constrained_dofs.size() << " Dirichlet BCs for solid mechanics");
}

} // namespace mpfem

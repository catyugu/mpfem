/**
 * @file linear_form.cpp
 * @brief Implementation of LinearForm class
 */

#include "linear_form.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

LinearForm::LinearForm(const DoFHandler* dof_handler)
    : dof_handler_(dof_handler)
    , mesh_(dof_handler ? dof_handler->fe_space()->mesh() : nullptr)
    , fe_space_(dof_handler ? dof_handler->fe_space() : nullptr)
    , update_flags_(UpdateFlags::UpdateDefault)
{
}

void LinearForm::assemble(LocalVectorAssembler local_assembler,
                         DynamicVector& vector) {
    if (!dof_handler_ || !mesh_ || !fe_space_) {
        MPFEM_ERROR("LinearForm: DoFHandler not initialized");
        return;
    }
    
    Index n_dofs = dof_handler_->n_dofs();
    vector.setZero(n_dofs);
    
    DynamicVector local_vector;
    std::vector<Index> local_dofs;
    
    Index global_cell_idx = 0;
    for (SizeType block_idx = 0; block_idx < mesh_->num_cell_blocks(); ++block_idx) {
        const auto& block = mesh_->cell_blocks()[block_idx];
        
        const FiniteElement* fe = fe_space_->get_fe(global_cell_idx);
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        FEValues fe_values(fe, update_flags_);
        int n_local = fe->dofs_per_cell();
        local_vector.setZero(n_local);
        
        for (SizeType e = 0; e < block.size(); ++e) {
            fe_values.reinit(*mesh_, global_cell_idx);
            dof_handler_->get_cell_dofs(global_cell_idx, local_dofs);
            
            local_vector.setZero();
            local_assembler(global_cell_idx, fe_values, local_vector);
            
            // Assemble into global vector
            for (int i = 0; i < n_local; ++i) {
                Index gi = local_dofs[i];
                if (gi >= n_dofs) continue;
                if (dof_handler_->is_constrained(gi)) continue;
                
                vector[gi] += local_vector[i];
            }
            
            ++global_cell_idx;
        }
    }
}

void LinearForm::assemble_with_source(
    LocalVectorAssembler local_assembler,
    DynamicVector& vector,
    const std::unordered_map<Index, Scalar>& sources) {
    
    if (!dof_handler_ || !mesh_ || !fe_space_) {
        MPFEM_ERROR("LinearForm: DoFHandler not initialized");
        return;
    }
    
    Index n_dofs = dof_handler_->n_dofs();
    vector.setZero(n_dofs);
    
    DynamicVector local_vector;
    std::vector<Index> local_dofs;
    
    Index global_cell_idx = 0;
    for (SizeType block_idx = 0; block_idx < mesh_->num_cell_blocks(); ++block_idx) {
        const auto& block = mesh_->cell_blocks()[block_idx];
        
        const FiniteElement* fe = fe_space_->get_fe(global_cell_idx);
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        FEValues fe_values(fe, update_flags_);
        int n_local = fe->dofs_per_cell();
        local_vector.setZero(n_local);
        
        for (SizeType e = 0; e < block.size(); ++e) {
            // Get domain ID for this cell
            Index domain_id = block.entity_id(e);
            
            // Get source for this domain
            Scalar source_val = 0.0;
            auto it = sources.find(domain_id);
            if (it != sources.end()) {
                source_val = it->second;
            }
            
            if (std::abs(source_val) < 1e-15) {
                ++global_cell_idx;
                continue;  // Skip if no source
            }
            
            fe_values.reinit(*mesh_, global_cell_idx);
            dof_handler_->get_cell_dofs(global_cell_idx, local_dofs);
            
            local_vector.setZero();
            local_assembler(global_cell_idx, fe_values, local_vector);
            
            // Scale by source value
            local_vector *= source_val;
            
            for (int i = 0; i < n_local; ++i) {
                Index gi = local_dofs[i];
                if (gi >= n_dofs) continue;
                if (dof_handler_->is_constrained(gi)) continue;
                
                vector[gi] += local_vector[i];
            }
            
            ++global_cell_idx;
        }
    }
}

void LinearForm::assemble_boundary(Index boundary_id,
                                  LocalVectorAssembler local_assembler,
                                  DynamicVector& vector) {
    if (!dof_handler_ || !mesh_ || !fe_space_) {
        MPFEM_ERROR("LinearForm: DoFHandler not initialized");
        return;
    }
    
    // Get faces belonging to this boundary
    const auto& face_topos = mesh_->face_topologies();
    
    DynamicVector local_vector;
    std::vector<Index> local_dofs;
    
    Index global_face_idx = 0;
    for (SizeType block_idx = 0; block_idx < mesh_->num_face_blocks(); ++block_idx) {
        const auto& block = mesh_->face_blocks()[block_idx];
        
        // Need to get FE from adjacent cell
        for (SizeType e = 0; e < block.size(); ++e) {
            // Check if this face belongs to the specified boundary
            if (global_face_idx >= static_cast<Index>(face_topos.size())) {
                ++global_face_idx;
                continue;
            }
            
            const auto& face_topo = face_topos[global_face_idx];
            if (!face_topo.is_boundary || face_topo.boundary_entity_id != boundary_id) {
                ++global_face_idx;
                continue;
            }
            
            // Get the adjacent cell
            Index cell_id = face_topo.cell_id;
            int local_face = face_topo.local_face_index;
            
            const FiniteElement* fe = fe_space_->get_fe(cell_id);
            if (!fe) {
                ++global_face_idx;
                continue;
            }
            
            // Create FEValues for face integration
            FEValues fe_values(fe, update_flags_ | UpdateFlags::UpdateNormals);
            fe_values.reinit_face(*mesh_, global_face_idx, cell_id, local_face);
            
            dof_handler_->get_cell_dofs(cell_id, local_dofs);
            
            int n_local = fe->dofs_per_cell();
            local_vector.setZero(n_local);
            local_assembler(global_face_idx, fe_values, local_vector);
            
            Index n_dofs = dof_handler_->n_dofs();
            for (int i = 0; i < n_local; ++i) {
                Index gi = local_dofs[i];
                if (gi >= n_dofs) continue;
                if (dof_handler_->is_constrained(gi)) continue;
                
                vector[gi] += local_vector[i];
            }
            
            ++global_face_idx;
        }
    }
}

void LinearForm::assemble_boundary_with_value(
    Index boundary_id,
    Scalar value,
    LocalVectorAssembler local_assembler,
    DynamicVector& vector) {
    
    // Create a wrapper assembler that scales by value
    auto wrapped_assembler = [value, &local_assembler](
        Index cell_id, const FEValues& fe, DynamicVector& local_vec) {
        local_assembler(cell_id, fe, local_vec);
        local_vec *= value;
    };
    
    assemble_boundary(boundary_id, wrapped_assembler, vector);
}

// ============================================================
// Predefined linear forms
// ============================================================

namespace LinearForms {

LocalVectorAssembler source(Scalar source_val) {
    return [source_val](Index, const FEValues& fe, DynamicVector& F_local) {
        int n = fe.dofs_per_cell();
        F_local.setZero(n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            for (int i = 0; i < n; ++i) {
                F_local[i] += source_val * fe.shape_value(i, q) * jxw;
            }
        }
    };
}

LocalVectorAssembler source_function(SourceFunction source_func) {
    return [source_func](Index, const FEValues& fe, DynamicVector& F_local) {
        int n = fe.dofs_per_cell();
        F_local.setZero(n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            const auto& pt = fe.quadrature_point(q);
            
            Scalar f_val = source_func(pt.x(), pt.y(), pt.z());
            
            for (int i = 0; i < n; ++i) {
                F_local[i] += f_val * fe.shape_value(i, q) * jxw;
            }
        }
    };
}

LocalVectorAssembler neumann_flux(Scalar flux) {
    return [flux](Index, const FEValues& fe, DynamicVector& F_local) {
        int n = fe.dofs_per_cell();
        F_local.setZero(n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            for (int i = 0; i < n; ++i) {
                F_local[i] += flux * fe.shape_value(i, q) * jxw;
            }
        }
    };
}

LocalVectorAssembler convection_rhs(Scalar h, Scalar T_inf) {
    return [h, T_inf](Index, const FEValues& fe, DynamicVector& F_local) {
        int n = fe.dofs_per_cell();
        F_local.setZero(n);
        
        Scalar h_Tinf = h * T_inf;
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            for (int i = 0; i < n; ++i) {
                F_local[i] += h_Tinf * fe.shape_value(i, q) * jxw;
            }
        }
    };
}

LocalVectorAssembler thermal_strain(Scalar alpha, Scalar delta_T,
                                   Scalar E, Scalar nu, int dim) {
    return [alpha, delta_T, E, nu, dim](Index, const FEValues& fe, DynamicVector& F_local) {
        int n_comp = fe.n_components();
        int n = fe.dofs_per_cell();
        
        if (n_comp != dim) {
            return;
        }
        
        F_local.setZero(n);
        
        // Elasticity parameters
        Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Scalar mu = E / (2.0 * (1.0 + nu));
        
        // Thermal strain
        Scalar eps_th = alpha * delta_T;
        
        // Thermal stress (for plane strain / 3D)
        Scalar sigma_th = -(3 * lambda + 2 * mu) * eps_th;  // Hydrostatic stress
        
        int n_strain = (dim == 3) ? 6 : 3;
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            int n_nodes = n / dim;
            
            for (int a = 0; a < n_nodes; ++a) {
                const auto& grad_a = fe.shape_grad(a, q);
                
                if (dim == 3) {
                    // For 3D: thermal strain is [eps_th, eps_th, eps_th, 0, 0, 0]
                    // Thermal stress: sigma_th = -E * alpha * dT / (1-2nu) for each normal component
                    Scalar coeff = (3 * lambda + 2 * mu) * eps_th * jxw;
                    
                    // F = int(B^T * sigma_th)
                    // sigma_th = [sigma_th, sigma_th, sigma_th, 0, 0, 0]
                    // B_a^T * sigma_th = [sigma_th * dN/dx, sigma_th * dN/dy, sigma_th * dN/dz]
                    
                    F_local[a*dim + 0] -= coeff * grad_a.x();
                    F_local[a*dim + 1] -= coeff * grad_a.y();
                    F_local[a*dim + 2] -= coeff * grad_a.z();
                } else {
                    // 2D plane strain
                    Scalar coeff = (2 * lambda + 2 * mu) * eps_th * jxw;
                    
                    F_local[a*dim + 0] -= coeff * grad_a.x();
                    F_local[a*dim + 1] -= coeff * grad_a.y();
                }
            }
        }
    };
}

}  // namespace LinearForms

}  // namespace mpfem

/**
 * @file bilinear_form.cpp
 * @brief Implementation of BilinearForm class
 */

#include "bilinear_form.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

BilinearForm::BilinearForm(const DoFHandler* dof_handler)
    : dof_handler_(dof_handler)
    , mesh_(dof_handler ? dof_handler->fe_space()->mesh() : nullptr)
    , fe_space_(dof_handler ? dof_handler->fe_space() : nullptr)
    , update_flags_(UpdateFlags::UpdateDefault)
    , n_entries_(0)
{
}

void BilinearForm::assemble(LocalMatrixAssembler local_assembler,
                            SparseMatrix& matrix,
                            bool symmetrize) {
    if (!dof_handler_ || !mesh_ || !fe_space_) {
        MPFEM_ERROR("BilinearForm: DoFHandler not initialized");
        return;
    }
    
    Index n_dofs = dof_handler_->n_dofs();
    
    // Use triplet list for efficient assembly
    std::vector<Eigen::Triplet<Scalar>> triplets;
    
    // Preallocate (estimate upper bound)
    size_t n_cells = mesh_->num_cells();
    int dofs_per_cell = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        dofs_per_cell = std::max(dofs_per_cell, 
            fe_space_->dofs_per_cell(to_geometry_type(block.type())));
    }
    triplets.reserve(n_cells * dofs_per_cell * dofs_per_cell);
    
    // Local data
    DynamicMatrix local_matrix;
    std::vector<Index> local_dofs;
    
    // Iterate over all cells
    Index global_cell_idx = 0;
    for (SizeType block_idx = 0; block_idx < mesh_->num_cell_blocks(); ++block_idx) {
        const auto& block = mesh_->cell_blocks()[block_idx];
        GeometryType geom_type = to_geometry_type(block.type());
        
        // Get FE for this geometry type
        const FiniteElement* fe = fe_space_->get_fe(global_cell_idx);
        if (!fe) {
            MPFEM_WARN("BilinearForm: No FE for cell " << global_cell_idx);
            global_cell_idx += block.size();
            continue;
        }
        
        // Create FEValues for this element type
        FEValues fe_values(fe, update_flags_);
        int n_local = fe->dofs_per_cell();
        local_matrix.setZero(n_local, n_local);
        
        for (SizeType e = 0; e < block.size(); ++e) {
            // Reinitialize FEValues for this cell
            fe_values.reinit(*mesh_, global_cell_idx);
            
            // Get local DoF indices
            dof_handler_->get_cell_dofs(global_cell_idx, local_dofs);
            
            // Compute local matrix via callback
            local_matrix.setZero();
            local_assembler(global_cell_idx, fe_values, local_matrix);
            
            // Assemble into global matrix
            for (int i = 0; i < n_local; ++i) {
                Index gi = local_dofs[i];
                if (gi >= n_dofs) continue;
                
                for (int j = 0; j < n_local; ++j) {
                    Index gj = local_dofs[j];
                    if (gj >= n_dofs) continue;
                    
                    // Skip constrained DoFs in row
                    if (dof_handler_->is_constrained(gi)) continue;
                    
                    triplets.emplace_back(gi, gj, local_matrix(i, j));
                }
            }
            
            ++global_cell_idx;
        }
    }
    
    // Build sparse matrix from triplets
    matrix.setZero();
    matrix.resize(n_dofs, n_dofs);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    // Symmetrize if requested
    if (symmetrize) {
        SparseMatrix Kt = matrix.transpose();
        matrix = 0.5 * (matrix + Kt);
    }
    
    n_entries_ = triplets.size();
    MPFEM_INFO("BilinearForm: Assembled " << n_entries_ << " entries into "
               << n_dofs << " x " << n_dofs << " matrix");
}

void BilinearForm::assemble_with_coefficients(
    LocalMatrixAssembler local_assembler,
    SparseMatrix& matrix,
    const std::unordered_map<Index, Scalar>& coefficients,
    bool symmetrize) {
    
    // Similar to assemble, but scale by domain coefficient
    if (!dof_handler_ || !mesh_ || !fe_space_) {
        MPFEM_ERROR("BilinearForm: DoFHandler not initialized");
        return;
    }
    
    Index n_dofs = dof_handler_->n_dofs();
    std::vector<Eigen::Triplet<Scalar>> triplets;
    
    size_t n_cells = mesh_->num_cells();
    int dofs_per_cell = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        dofs_per_cell = std::max(dofs_per_cell,
            fe_space_->dofs_per_cell(to_geometry_type(block.type())));
    }
    triplets.reserve(n_cells * dofs_per_cell * dofs_per_cell);
    
    DynamicMatrix local_matrix;
    std::vector<Index> local_dofs;
    
    Index global_cell_idx = 0;
    for (SizeType block_idx = 0; block_idx < mesh_->num_cell_blocks(); ++block_idx) {
        const auto& block = mesh_->cell_blocks()[block_idx];
        GeometryType geom_type = to_geometry_type(block.type());
        
        const FiniteElement* fe = fe_space_->get_fe(global_cell_idx);
        if (!fe) {
            global_cell_idx += block.size();
            continue;
        }
        
        FEValues fe_values(fe, update_flags_);
        int n_local = fe->dofs_per_cell();
        local_matrix.setZero(n_local, n_local);
        
        for (SizeType e = 0; e < block.size(); ++e) {
            // Get domain ID for this cell
            Index domain_id = block.entity_id(e);
            
            // Get coefficient for this domain
            Scalar coeff = 1.0;
            auto it = coefficients.find(domain_id);
            if (it != coefficients.end()) {
                coeff = it->second;
            }
            
            fe_values.reinit(*mesh_, global_cell_idx);
            dof_handler_->get_cell_dofs(global_cell_idx, local_dofs);
            
            local_matrix.setZero();
            local_assembler(global_cell_idx, fe_values, local_matrix);
            
            // Scale by domain coefficient
            local_matrix *= coeff;
            
            for (int i = 0; i < n_local; ++i) {
                Index gi = local_dofs[i];
                if (gi >= n_dofs) continue;
                
                for (int j = 0; j < n_local; ++j) {
                    Index gj = local_dofs[j];
                    if (gj >= n_dofs) continue;
                    
                    if (dof_handler_->is_constrained(gi)) continue;
                    
                    triplets.emplace_back(gi, gj, local_matrix(i, j));
                }
            }
            
            ++global_cell_idx;
        }
    }
    
    matrix.setZero();
    matrix.resize(n_dofs, n_dofs);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    if (symmetrize) {
        SparseMatrix Kt = matrix.transpose();
        matrix = 0.5 * (matrix + Kt);
    }
    
    n_entries_ = triplets.size();
    MPFEM_INFO("BilinearForm: Assembled " << n_entries_ << " entries with domain coefficients");
}

// ============================================================
// Predefined bilinear forms
// ============================================================

namespace BilinearForms {

LocalMatrixAssembler laplacian(Scalar conductivity) {
    return [conductivity](Index, const FEValues& fe, DynamicMatrix& K_local) {
        int n = fe.dofs_per_cell();
        K_local.setZero(n, n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q) * conductivity;
            
            for (int i = 0; i < n; ++i) {
                const auto& grad_i = fe.shape_grad(i, q);
                
                for (int j = 0; j < n; ++j) {
                    const auto& grad_j = fe.shape_grad(j, q);
                    K_local(i, j) += grad_i.dot(grad_j) * jxw;
                }
            }
        }
    };
}

LocalMatrixAssembler laplacian_anisotropic(const Tensor<2, 3>& K_tensor) {
    return [K_tensor](Index, const FEValues& fe, DynamicMatrix& K_local) {
        int n = fe.dofs_per_cell();
        K_local.setZero(n, n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            for (int i = 0; i < n; ++i) {
                const auto& grad_i = fe.shape_grad(i, q);
                // K * grad_i
                Tensor<1, 3> K_grad_i = K_tensor * grad_i;
                
                for (int j = 0; j < n; ++j) {
                    const auto& grad_j = fe.shape_grad(j, q);
                    K_local(i, j) += K_grad_i.dot(grad_j) * jxw;
                }
            }
        }
    };
}

LocalMatrixAssembler mass(Scalar coefficient) {
    return [coefficient](Index, const FEValues& fe, DynamicMatrix& M_local) {
        int n = fe.dofs_per_cell();
        M_local.setZero(n, n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q) * coefficient;
            
            for (int i = 0; i < n; ++i) {
                Scalar N_i = fe.shape_value(i, q);
                
                for (int j = 0; j < n; ++j) {
                    Scalar N_j = fe.shape_value(j, q);
                    M_local(i, j) += N_i * N_j * jxw;
                }
            }
        }
    };
}

LocalMatrixAssembler elasticity(Scalar E, Scalar nu, int dim) {
    return [E, nu, dim](Index, const FEValues& fe, DynamicMatrix& K_local) {
        int n_comp = fe.n_components();
        int n = fe.dofs_per_cell();
        
        if (n_comp != dim) {
            MPFEM_ERROR("Elasticity: FE components (" << n_comp 
                       << ") != dimension (" << dim << ")");
            return;
        }
        
        K_local.setZero(n, n);
        
        // Isotropic elasticity tensor for plane strain / 3D
        Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Scalar mu = E / (2.0 * (1.0 + nu));
        
        // For 2D plane strain, use the same formula (different for plane stress)
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            
            // B matrix for each DoF: strain-displacement matrix
            // For 3D: strain = [e11, e22, e33, 2e12, 2e23, 2e13]
            // B_i = [[dNi/dx, 0, 0],
            //        [0, dNi/dy, 0],
            //        [0, 0, dNi/dz],
            //        [dNi/dy, dNi/dx, 0],
            //        [0, dNi/dz, dNi/dy],
            //        [dNi/dz, 0, dNi/dx]]
            
            int n_strain = (dim == 3) ? 6 : 3;
            
            // Compute B matrices for each node
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> B;
            int n_nodes = n / dim;
            B.resize(n_nodes);
            
            for (int a = 0; a < n_nodes; ++a) {
                B[a].setZero(n_strain, dim);
                
                const auto& grad_a = fe.shape_grad(a, q);
                Scalar dNx = grad_a.x();
                Scalar dNy = grad_a.y();
                Scalar dNz = grad_a.z();
                
                if (dim == 3) {
                    B[a] << dNx,  0,   0,
                            0,  dNy,   0,
                            0,   0,  dNz,
                           dNy, dNx,   0,
                            0,  dNz, dNy,
                           dNz,   0, dNx;
                } else {
                    // 2D plane strain
                    B[a] << dNx,   0,
                            0,   dNy,
                           dNy, dNx;
                }
            }
            
            // Compute D matrix (constitutive)
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> D;
            D.setZero(n_strain, n_strain);
            
            if (dim == 3) {
                // 3D isotropic elasticity
                D(0,0) = D(1,1) = D(2,2) = lambda + 2*mu;
                D(0,1) = D(0,2) = D(1,0) = D(1,2) = D(2,0) = D(2,1) = lambda;
                D(3,3) = D(4,4) = D(5,5) = mu;
            } else {
                // 2D plane strain
                D(0,0) = D(1,1) = lambda + 2*mu;
                D(0,1) = D(1,0) = lambda;
                D(2,2) = mu;
            }
            
            // Assemble element stiffness matrix
            for (int a = 0; a < n_nodes; ++a) {
                for (int b = 0; b < n_nodes; ++b) {
                    // K_ab = B_a^T * D * B_b * jxw
                    auto K_ab = B[a].transpose() * D * B[b] * jxw;
                    
                    // Copy to local matrix
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            K_local(a*dim + i, b*dim + j) = K_ab(i, j);
                        }
                    }
                }
            }
        }
    };
}

LocalMatrixAssembler convection_bc(Scalar h) {
    return [h](Index, const FEValues& fe, DynamicMatrix& K_local) {
        int n = fe.dofs_per_cell();
        K_local.setZero(n, n);
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q) * h;
            
            for (int i = 0; i < n; ++i) {
                Scalar N_i = fe.shape_value(i, q);
                
                for (int j = 0; j < n; ++j) {
                    Scalar N_j = fe.shape_value(j, q);
                    K_local(i, j) += N_i * N_j * jxw;
                }
            }
        }
    };
}

}  // namespace BilinearForms

}  // namespace mpfem

/**
 * @file bilinear_form.cpp
 * @brief Implementation of BilinearForm class
 */

#include "bilinear_form.hpp"
#include "fe_values.hpp"
#include "fem/fe_collection.hpp"
#include "core/logger.hpp"
#include "mesh/element.hpp"
#include <algorithm>

namespace mpfem {

BilinearForm::BilinearForm(const FieldSpace* field)
    : field_(field)
    , mesh_(field ? field->mesh() : nullptr)
    , n_entries_(0)
{
}

void BilinearForm::assemble(LocalMatrixAssembler local_assembler,
                            SparseMatrix& matrix,
                            bool symmetrize) {
    if (!field_ || !mesh_) {
        MPFEM_ERROR("BilinearForm: Field not initialized");
        return;
    }
    
    Index n_dofs = field_->n_dofs();
    matrix.resize(n_dofs, n_dofs);
    matrix.setZero();
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(mesh_->num_cells() * 20);
    
    DynamicMatrix local_matrix;
    
    // 遍历所有单元块，根据单元类型选择对应的 FE
    Index cell_id = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        auto fe = create_fe(geom_type, field_->order(), field_->n_components());
        if (!fe) {
            MPFEM_WARN("No FE for geometry type " << static_cast<int>(geom_type) 
                       << ", skipping " << block.size() << " cells");
            cell_id += static_cast<Index>(block.size());
            continue;
        }
        
        FEValues fe_values(fe.get(), update_flags_);
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_id) {
            fe_values.reinit(*field_, cell_id);
            
            int n_local = fe_values.dofs_per_cell();
            local_matrix.setZero(n_local, n_local);
            
            local_assembler(cell_id, fe_values, local_matrix);
            
            // 直接组装到 triplets
            std::vector<Index> cell_dofs;
            field_->get_cell_dofs(cell_id, cell_dofs);
            
            for (int i = 0; i < n_local; ++i) {
                for (int j = 0; j < n_local; ++j) {
                    if (std::abs(local_matrix(i, j)) > 1e-15) {
                        triplets.emplace_back(cell_dofs[i], cell_dofs[j], local_matrix(i, j));
                    }
                }
            }
        }
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    if (symmetrize) {
        SparseMatrix Kt = matrix.transpose();
        matrix = 0.5 * (matrix + Kt);
    }
    
    matrix.makeCompressed();
    n_entries_ = matrix.nonZeros();
    MPFEM_INFO("BilinearForm: Assembled " << n_entries_ << " non-zeros");
}

void BilinearForm::assemble_with_coefficients(
    LocalMatrixAssembler local_assembler,
    SparseMatrix& matrix,
    const std::unordered_map<Index, Scalar>& coefficients,
    bool symmetrize) {
    
    if (!field_ || !mesh_) return;
    
    Index n_dofs = field_->n_dofs();
    matrix.resize(n_dofs, n_dofs);
    matrix.setZero();
    
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(mesh_->num_cells() * 20);
    
    DynamicMatrix local_matrix;
    
    Index cell_id = 0;
    for (const auto& block : mesh_->cell_blocks()) {
        GeometryType geom_type = to_geometry_type(block.type());
        auto fe = create_fe(geom_type, field_->order(), field_->n_components());
        if (!fe) {
            cell_id += static_cast<Index>(block.size());
            continue;
        }
        
        FEValues fe_values(fe.get(), update_flags_);
        
        for (SizeType e = 0; e < block.size(); ++e, ++cell_id) {
            Index domain_id = mesh_->get_cell_domain_id(cell_id);
            
            Scalar coeff = 1.0;
            auto it = coefficients.find(domain_id);
            if (it != coefficients.end()) {
                coeff = it->second;
            }
            
            fe_values.reinit(*field_, cell_id);
            
            int n_local = fe_values.dofs_per_cell();
            local_matrix.setZero(n_local, n_local);
            
            local_assembler(cell_id, fe_values, local_matrix);
            local_matrix *= coeff;
            
            std::vector<Index> cell_dofs;
            field_->get_cell_dofs(cell_id, cell_dofs);
            
            for (int i = 0; i < n_local; ++i) {
                for (int j = 0; j < n_local; ++j) {
                    if (std::abs(local_matrix(i, j)) > 1e-15) {
                        triplets.emplace_back(cell_dofs[i], cell_dofs[j], local_matrix(i, j));
                    }
                }
            }
        }
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    
    if (symmetrize) {
        SparseMatrix Kt = matrix.transpose();
        matrix = 0.5 * (matrix + Kt);
    }
    
    matrix.makeCompressed();
    n_entries_ = matrix.nonZeros();
}

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
                    K_local(i, j) += grad_i.dot(fe.shape_grad(j, q)) * jxw;
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
                Tensor<1, 3> K_grad_i = K_tensor * grad_i;
                
                for (int j = 0; j < n; ++j) {
                    K_local(i, j) += K_grad_i.dot(fe.shape_grad(j, q)) * jxw;
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
                    M_local(i, j) += N_i * fe.shape_value(j, q) * jxw;
                }
            }
        }
    };
}

LocalMatrixAssembler elasticity(Scalar E, Scalar nu, int dim) {
    return [E, nu, dim](Index, const FEValues& fe, DynamicMatrix& K_local) {
        int n = fe.dofs_per_cell();
        K_local.setZero(n, n);
        
        Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Scalar mu = E / (2.0 * (1.0 + nu));
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            int n_nodes = n / dim;
            
            for (int a = 0; a < n_nodes; ++a) {
                const auto& grad_a = fe.shape_grad(a, q);
                for (int b = 0; b < n_nodes; ++b) {
                    const auto& grad_b = fe.shape_grad(b, q);
                    
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
                    K_local(i, j) += N_i * fe.shape_value(j, q) * jxw;
                }
            }
        }
    };
}

}  // namespace BilinearForms

}  // namespace mpfem
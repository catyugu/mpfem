/**
 * @file linear_form.cpp
 * @brief Implementation of LinearForm class
 */

#include "linear_form.hpp"
#include "fe_values.hpp"
#include "fem/fe_collection.hpp"
#include "core/logger.hpp"
#include "mesh/element.hpp"
#include <algorithm>

namespace mpfem {

LinearForm::LinearForm(const FieldSpace* field)
    : field_(field)
    , mesh_(field ? field->mesh() : nullptr)
{
}

void LinearForm::assemble(LocalVectorAssembler local_assembler, DynamicVector& vector) {
    if (!field_ || !mesh_) {
        MPFEM_ERROR("LinearForm: Field not initialized");
        return;
    }
    
    Index n_dofs = field_->n_dofs();
    vector.setZero(n_dofs);
    
    DynamicVector local_vector;
    
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
            fe_values.reinit(*field_, cell_id);
            
            int n_local = fe_values.dofs_per_cell();
            local_vector.setZero(n_local);
            local_assembler(cell_id, fe_values, local_vector);
            
            std::vector<Index> cell_dofs;
            field_->get_cell_dofs(cell_id, cell_dofs);
            
            for (int i = 0; i < n_local; ++i) {
                vector[cell_dofs[i]] += local_vector(i);
            }
        }
    }
}

void LinearForm::assemble_with_source(
    LocalVectorAssembler local_assembler,
    DynamicVector& vector,
    const std::unordered_map<Index, Scalar>& sources) {
    
    if (!field_ || !mesh_) return;
    
    Index n_dofs = field_->n_dofs();
    vector.setZero(n_dofs);
    
    DynamicVector local_vector;
    
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
            
            Scalar source_val = 0.0;
            auto it = sources.find(domain_id);
            if (it != sources.end()) {
                source_val = it->second;
            }
            
            if (std::abs(source_val) < 1e-15) continue;
            
            fe_values.reinit(*field_, cell_id);
            
            int n_local = fe_values.dofs_per_cell();
            local_vector.setZero(n_local);
            local_assembler(cell_id, fe_values, local_vector);
            local_vector *= source_val;
            
            std::vector<Index> cell_dofs;
            field_->get_cell_dofs(cell_id, cell_dofs);
            
            for (int i = 0; i < n_local; ++i) {
                vector[cell_dofs[i]] += local_vector(i);
            }
        }
    }
}

void LinearForm::assemble_boundary(Index boundary_id,
                                  LocalVectorAssembler local_assembler,
                                  DynamicVector& vector) {
    if (!field_ || !mesh_) return;
    
    const auto& face_topos = mesh_->face_topologies();
    
    DynamicVector local_vector;
    
    for (Index face_id = 0; face_id < static_cast<Index>(face_topos.size()); ++face_id) {
        const auto& face_topo = face_topos[face_id];
        if (!face_topo.is_boundary || face_topo.boundary_entity_id != boundary_id) continue;
        
        Index cell_id = face_topo.cell_id;
        int local_face = face_topo.local_face_index;
        
        // 获取单元的几何类型
        GeometryType geom_type = GeometryType::Invalid;
        Index remaining = cell_id;
        for (const auto& block : mesh_->cell_blocks()) {
            if (remaining < static_cast<Index>(block.size())) {
                geom_type = to_geometry_type(block.type());
                break;
            }
            remaining -= static_cast<Index>(block.size());
        }
        
        auto fe = create_fe(geom_type, field_->order(), field_->n_components());
        if (!fe) continue;
        
        FEValues fe_values(fe.get(), update_flags_ | UpdateFlags::UpdateNormals);
        fe_values.reinit_face(*field_, face_id, cell_id, local_face);
        
        int n_local = fe_values.dofs_per_cell();
        local_vector.setZero(n_local);
        local_assembler(face_id, fe_values, local_vector);
        
        std::vector<Index> cell_dofs;
        field_->get_cell_dofs(cell_id, cell_dofs);
        
        for (int i = 0; i < n_local; ++i) {
            vector[cell_dofs[i]] += local_vector(i);
        }
    }
}

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
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            for (int i = 0; i < n; ++i) {
                F_local[i] += h * T_inf * fe.shape_value(i, q) * jxw;
            }
        }
    };
}

LocalVectorAssembler thermal_strain(Scalar alpha, Scalar delta_T,
                                   Scalar E, Scalar nu, int dim) {
    return [alpha, delta_T, E, nu, dim](Index, const FEValues& fe, DynamicVector& F_local) {
        int n_comp = fe.n_components();
        int n = fe.dofs_per_cell();
        if (n_comp != dim) return;
        
        F_local.setZero(n);
        
        Scalar lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Scalar mu = E / (2.0 * (1.0 + nu));
        Scalar eps_th = alpha * delta_T;
        
        for (int q = 0; q < fe.n_quadrature_points(); ++q) {
            Scalar jxw = fe.JxW(q);
            int n_nodes = n / dim;
            
            for (int a = 0; a < n_nodes; ++a) {
                const auto& grad_a = fe.shape_grad(a, q);
                Scalar coeff = (dim == 3) ? 
                    (3 * lambda + 2 * mu) * eps_th * jxw :
                    (2 * lambda + 2 * mu) * eps_th * jxw;
                
                for (int d = 0; d < dim; ++d) {
                    F_local[a*dim + d] -= coeff * grad_a[d];
                }
            }
        }
    };
}

}  // namespace LinearForms

}  // namespace mpfem
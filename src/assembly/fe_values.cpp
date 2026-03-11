/**
 * @file fe_values.cpp
 * @brief Implementation of FEValues class
 */

#include "fe_values.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

FEValues::FEValues(const FiniteElement* fe, UpdateFlags flags)
    : fe_(fe)
    , flags_(flags)
    , n_qpoints_(fe ? fe->n_quadrature_points() : 0)
    , dofs_per_cell_(fe ? fe->dofs_per_cell() : 0)
    , n_components_(fe ? fe->n_components() : 1)
    , is_face_(false)
    , current_face_id_(InvalidIndex)
    , current_local_face_(-1)
{
    if (fe_) {
        qp_data_.resize(n_qpoints_);
        for (auto& qpd : qp_data_) {
            qpd.shape_values.resize(dofs_per_cell_);
            qpd.shape_gradients.resize(dofs_per_cell_);
        }
    }
}

void FEValues::reinit(const FieldSpace& field, Index cell_id) {
    current_field_ = &field;
    current_cell_id_ = cell_id;
    is_face_ = false;
    current_face_id_ = InvalidIndex;
    current_local_face_ = -1;
    
    // Get cell DoFs (using FieldSpace's low-level API - appropriate for framework code)
    field.get_cell_dofs(cell_id, cell_dofs_);
    
    // Clear cached field DoFs for new cell
    cached_field_dofs_.clear();
    
    compute_cell_data(*field.mesh(), cell_id);
}

void FEValues::reinit_face(const FieldSpace& field, Index face_id, Index cell_id, int local_face_index) {
    current_field_ = &field;
    current_cell_id_ = cell_id;
    is_face_ = true;
    current_face_id_ = face_id;
    current_local_face_ = local_face_index;
    
    field.get_cell_dofs(cell_id, cell_dofs_);
    
    // Clear cached field DoFs for new cell
    cached_field_dofs_.clear();
    
    compute_face_data(*field.mesh(), face_id, cell_id, local_face_index);
}

const std::vector<Index>& FEValues::get_cached_field_dofs(const FieldID& field_name) const {
    auto it = cached_field_dofs_.find(field_name);
    if (it != cached_field_dofs_.end()) {
        return it->second;
    }
    
    // Fetch and cache
    const FieldSpace* field = field_registry_ ? field_registry_->get_field(field_name) : nullptr;
    if (!field) {
        static const std::vector<Index> empty;
        return empty;
    }
    
    auto& dofs = cached_field_dofs_[field_name];
    field->get_cell_dofs(current_cell_id_, dofs);
    return dofs;
}

void FEValues::assemble_local_to_global(SparseMatrix& K, const DynamicMatrix& local_K) const {
    if (!current_field_ || cell_dofs_.empty()) return;
    
    int n = static_cast<int>(cell_dofs_.size());
    for (int i = 0; i < n; ++i) {
        Index dof_i = cell_dofs_[i];
        for (int j = 0; j < n; ++j) {
            Index dof_j = cell_dofs_[j];
            K.coeffRef(dof_i, dof_j) += local_K(i, j);
        }
    }
}

void FEValues::assemble_rhs(DynamicVector& f, const DynamicVector& local_f) const {
    if (!current_field_ || cell_dofs_.empty()) return;
    
    int n = static_cast<int>(cell_dofs_.size());
    for (int i = 0; i < n; ++i) {
        f[cell_dofs_[i]] += local_f(i);
    }
}

Scalar FEValues::value(int q) const {
    if (!current_field_ || cell_dofs_.empty()) return 0.0;
    
    const DynamicVector& sol = current_field_->solution();
    Scalar val = 0.0;
    int n = static_cast<int>(cell_dofs_.size());
    
    for (int i = 0; i < n && i < dofs_per_cell_; ++i) {
        if (cell_dofs_[i] < sol.size()) {
            val += qp_data_[q].shape_values[i] * sol[cell_dofs_[i]];
        }
    }
    return val;
}

Tensor<1, 3> FEValues::gradient(int q) const {
    if (!current_field_ || cell_dofs_.empty()) return Tensor<1, 3>::Zero();
    
    const DynamicVector& sol = current_field_->solution();
    Tensor<1, 3> grad = Tensor<1, 3>::Zero();
    int n = static_cast<int>(cell_dofs_.size());
    
    for (int i = 0; i < n && i < dofs_per_cell_; ++i) {
        if (cell_dofs_[i] < sol.size()) {
            grad += qp_data_[q].shape_gradients[i] * sol[cell_dofs_[i]];
        }
    }
    return grad;
}

void FEValues::values(std::vector<Scalar>& vals) const {
    vals.resize(n_qpoints_, 0.0);
    for (int q = 0; q < n_qpoints_; ++q) {
        vals[q] = value(q);
    }
}

void FEValues::gradients(std::vector<Tensor<1, 3>>& grads) const {
    grads.resize(n_qpoints_, Tensor<1, 3>::Zero());
    for (int q = 0; q < n_qpoints_; ++q) {
        grads[q] = gradient(q);
    }
}

Tensor<1, 3> FEValues::vector_value(int q) const {
    if (!current_field_ || cell_dofs_.empty()) return Tensor<1, 3>::Zero();
    
    const DynamicVector& sol = current_field_->solution();
    int n_comp = current_field_->n_components();
    int n_nodes = static_cast<int>(cell_dofs_.size()) / n_comp;
    
    Tensor<1, 3> val = Tensor<1, 3>::Zero();
    
    for (int c = 0; c < n_comp && c < 3; ++c) {
        for (int i = 0; i < n_nodes && i < dofs_per_cell_; ++i) {
            Index dof = cell_dofs_[i * n_comp + c];
            if (dof < sol.size()) {
                val[c] += qp_data_[q].shape_values[i] * sol[dof];
            }
        }
    }
    return val;
}

Scalar FEValues::field_value(const FieldID& field_name, int q) const {
    if (!field_registry_) {
        MPFEM_WARN("Field registry not set, cannot query field '" << field_name << "'");
        return 0.0;
    }
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field) {
        MPFEM_WARN("Field '" << field_name << "' not found in registry");
        return 0.0;
    }
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return 0.0;
    
    const DynamicVector& sol = field->solution();
    Scalar value = 0.0;
    int n_shape = std::min(static_cast<int>(field_dofs.size()), dofs_per_cell_);
    
    for (int i = 0; i < n_shape; ++i) {
        if (field_dofs[i] < sol.size()) {
            value += qp_data_[q].shape_values[i] * sol[field_dofs[i]];
        }
    }
    
    return value;
}

void FEValues::field_values(const FieldID& field_name, std::vector<Scalar>& values) const {
    values.resize(n_qpoints_, 0.0);
    
    if (!field_registry_) {
        MPFEM_WARN("Field registry not set");
        return;
    }
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field) {
        MPFEM_WARN("Field '" << field_name << "' not found");
        return;
    }
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return;
    
    const DynamicVector& sol = field->solution();
    int n_shape = std::min(static_cast<int>(field_dofs.size()), dofs_per_cell_);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        Scalar val = 0.0;
        for (int i = 0; i < n_shape; ++i) {
            if (field_dofs[i] < sol.size()) {
                val += qp_data_[q].shape_values[i] * sol[field_dofs[i]];
            }
        }
        values[q] = val;
    }
}

Tensor<1, 3> FEValues::field_vector(const FieldID& field_name, int q) const {
    if (!field_registry_) return Tensor<1, 3>::Zero();
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field || field->type() != FieldType::Vector) return Tensor<1, 3>::Zero();
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return Tensor<1, 3>::Zero();
    
    const DynamicVector& sol = field->solution();
    int n_comp = field->n_components();
    int n_nodes_per_cell = field_dofs.size() / n_comp;
    
    Tensor<1, 3> value = Tensor<1, 3>::Zero();
    
    for (int c = 0; c < n_comp && c < 3; ++c) {
        for (int i = 0; i < n_nodes_per_cell && i < dofs_per_cell_; ++i) {
            Index dof = field_dofs[i * n_comp + c];
            if (dof < sol.size()) {
                value[c] += qp_data_[q].shape_values[i] * sol[dof];
            }
        }
    }
    
    return value;
}

void FEValues::field_vectors(const FieldID& field_name, std::vector<Tensor<1, 3>>& values) const {
    values.resize(n_qpoints_, Tensor<1, 3>::Zero());
    
    if (!field_registry_) return;
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field || field->type() != FieldType::Vector) return;
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return;
    
    const DynamicVector& sol = field->solution();
    int n_comp = field->n_components();
    int n_nodes_per_cell = field_dofs.size() / n_comp;
    
    for (int q = 0; q < n_qpoints_; ++q) {
        for (int c = 0; c < n_comp && c < 3; ++c) {
            for (int i = 0; i < n_nodes_per_cell && i < dofs_per_cell_; ++i) {
                Index dof = field_dofs[i * n_comp + c];
                if (dof < sol.size()) {
                    values[q][c] += qp_data_[q].shape_values[i] * sol[dof];
                }
            }
        }
    }
}

Tensor<1, 3> FEValues::field_gradient(const FieldID& field_name, int q) const {
    if (!field_registry_) return Tensor<1, 3>::Zero();
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field) return Tensor<1, 3>::Zero();
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return Tensor<1, 3>::Zero();
    
    const DynamicVector& sol = field->solution();
    Tensor<1, 3> grad = Tensor<1, 3>::Zero();
    int n_shape = std::min(static_cast<int>(field_dofs.size()), dofs_per_cell_);
    
    for (int i = 0; i < n_shape; ++i) {
        if (field_dofs[i] < sol.size()) {
            grad += qp_data_[q].shape_gradients[i] * sol[field_dofs[i]];
        }
    }
    
    return grad;
}

void FEValues::field_gradients(const FieldID& field_name, std::vector<Tensor<1, 3>>& gradients) const {
    gradients.resize(n_qpoints_, Tensor<1, 3>::Zero());
    
    if (!field_registry_) return;
    
    const FieldSpace* field = field_registry_->get_field(field_name);
    if (!field) return;
    
    // Use cached DoFs for performance
    const auto& field_dofs = get_cached_field_dofs(field_name);
    if (field_dofs.empty()) return;
    
    const DynamicVector& sol = field->solution();
    int n_shape = std::min(static_cast<int>(field_dofs.size()), dofs_per_cell_);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        for (int i = 0; i < n_shape; ++i) {
            if (field_dofs[i] < sol.size()) {
                gradients[q] += qp_data_[q].shape_gradients[i] * sol[field_dofs[i]];
            }
        }
    }
}

void FEValues::compute_cell_data(const Mesh& mesh, Index cell_id) {
    trans_.reinit(mesh, cell_id);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        const auto& ip = fe_->quadrature_point(q);
        trans_.set_reference_point(ip.coord);
        
        if (has_flag(flags_, UpdateFlags::UpdateJxW)) {
            qp_data_[q].JxW = trans_.det_jacobian() * ip.weight;
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateValues)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                qp_data_[q].shape_values[i] = fe_->shape_value(i, q);
            }
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateGradients)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                const Tensor<1, 3>& grad_ref = fe_->shape_gradient(i, q);
                qp_data_[q].shape_gradients[i] = trans_.transform_gradient(grad_ref);
            }
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateQuadraturePoints)) {
            qp_data_[q].physical_point = trans_.transform(ip.coord);
        }
    }
}

void FEValues::compute_face_data(const Mesh& mesh, Index face_id, Index cell_id, int local_face) {
    trans_.reinit(mesh, cell_id);
    
    GeometryType face_geom_type = GeometryType::Invalid;
    Index local_face_idx = 0;
    Index remaining = face_id;
    
    for (const auto& block : mesh.face_blocks()) {
        if (remaining < static_cast<Index>(block.size())) {
            face_geom_type = to_geometry_type(block.type());
            local_face_idx = remaining;
            break;
        }
        remaining -= static_cast<Index>(block.size());
    }
    
    const ElementBlock* face_block = nullptr;
    remaining = face_id;
    for (const auto& block : mesh.face_blocks()) {
        if (remaining < static_cast<Index>(block.size())) {
            face_block = &block;
            local_face_idx = remaining;
            break;
        }
        remaining -= static_cast<Index>(block.size());
    }
    
    std::vector<Point<3>> face_vertices;
    if (face_block) {
        auto verts = face_block->element_vertices(local_face_idx);
        face_vertices.resize(verts.size());
        for (size_t i = 0; i < verts.size(); ++i) {
            face_vertices[i] = mesh.vertex(verts[i]);
        }
    }
    
    Tensor<1, 3> face_normal = Tensor<1, 3>::Zero();
    if (face_vertices.size() >= 3) {
        Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
        Tensor<1, 3> e2 = face_vertices[2] - face_vertices[0];
        face_normal = e1.cross(e2).normalized();
    }
    
    for (int q = 0; q < n_qpoints_; ++q) {
        const auto& ip = fe_->quadrature_point(q);
        
        if (has_flag(flags_, UpdateFlags::UpdateJxW)) {
            Scalar face_detJ = 1.0;
            if (face_vertices.size() == 3) {
                Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
                Tensor<1, 3> e2 = face_vertices[2] - face_vertices[0];
                face_detJ = 0.5 * e1.cross(e2).norm();
            } else if (face_vertices.size() == 4) {
                Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
                Tensor<1, 3> e2 = face_vertices[3] - face_vertices[0];
                Tensor<1, 3> e3 = face_vertices[2] - face_vertices[1];
                Tensor<1, 3> e4 = face_vertices[2] - face_vertices[3];
                face_detJ = 0.25 * (e1.cross(e2).norm() + e3.cross(e4).norm());
            }
            
            if (face_geom_type == GeometryType::Triangle) {
                qp_data_[q].JxW = ip.weight * 2.0 * face_detJ;
            } else {
                qp_data_[q].JxW = ip.weight * face_detJ;
            }
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateValues)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                qp_data_[q].shape_values[i] = fe_->shape_value(i, q);
            }
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateGradients)) {
            trans_.set_reference_point(ip.coord);
            for (int i = 0; i < dofs_per_cell_; ++i) {
                const Tensor<1, 3>& grad_ref = fe_->shape_gradient(i, q);
                qp_data_[q].shape_gradients[i] = trans_.transform_gradient(grad_ref);
            }
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateNormals)) {
            qp_data_[q].normal = face_normal;
        }
        
        if (has_flag(flags_, UpdateFlags::UpdateQuadraturePoints)) {
            qp_data_[q].physical_point = trans_.transform(ip.coord);
        }
    }
}

}  // namespace mpfem

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

void FEValues::reinit(const Mesh& mesh, Index cell_id) {
    is_face_ = false;
    current_face_id_ = InvalidIndex;
    current_local_face_ = -1;
    
    compute_cell_data(mesh, cell_id);
}

void FEValues::reinit_face(const Mesh& mesh, Index face_id, Index cell_id, int local_face_index) {
    is_face_ = true;
    current_face_id_ = face_id;
    current_local_face_ = local_face_index;
    
    compute_face_data(mesh, face_id, cell_id, local_face_index);
}

void FEValues::compute_cell_data(const Mesh& mesh, Index cell_id) {
    // Initialize element transformation
    trans_.reinit(mesh, cell_id);
    
    // Get element type from cell block
    GeometryType geom_type = GeometryType::Invalid;
    Index local_cell_idx = 0;
    Index remaining = cell_id;
    
    for (const auto& block : mesh.cell_blocks()) {
        if (remaining < static_cast<Index>(block.size())) {
            geom_type = to_geometry_type(block.type());
            local_cell_idx = remaining;
            break;
        }
        remaining -= static_cast<Index>(block.size());
    }
    
    // Compute data at each quadrature point
    for (int q = 0; q < n_qpoints_; ++q) {
        const auto& ip = fe_->quadrature_point(q);
        
        // Update transformation to this quadrature point
        trans_.set_reference_point(ip.coord);
        
        // Compute JxW
        if (has_flag(flags_, UpdateFlags::UpdateJxW)) {
            qp_data_[q].JxW = trans_.det_jacobian() * ip.weight;
        }
        
        // Compute shape values (direct from FE)
        if (has_flag(flags_, UpdateFlags::UpdateValues)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                qp_data_[q].shape_values[i] = fe_->shape_value(i, q);
            }
        }
        
        // Compute physical gradients
        if (has_flag(flags_, UpdateFlags::UpdateGradients)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                // Get reference gradient
                const Tensor<1, 3>& grad_ref = fe_->shape_gradient(i, q);
                // Transform to physical gradient: grad_phys = inv(J)^T * grad_ref
                qp_data_[q].shape_gradients[i] = trans_.transform_gradient(grad_ref);
            }
        }
        
        // Compute physical quadrature point position
        if (has_flag(flags_, UpdateFlags::UpdateQuadraturePoints)) {
            qp_data_[q].physical_point = trans_.transform(ip.coord);
        }
    }
}

void FEValues::compute_face_data(const Mesh& mesh, Index face_id, Index cell_id, int local_face) {
    // First, initialize for the adjacent cell to get the Jacobian
    trans_.reinit(mesh, cell_id);
    
    // Get face block and geometry type
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
    
    // Get face vertices for normal computation
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
    
    // Compute face normal (for 3D elements, face is 2D)
    Tensor<1, 3> face_normal = Tensor<1, 3>::Zero();
    if (face_vertices.size() >= 3) {
        // Use cross product of first two edges
        Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
        Tensor<1, 3> e2 = face_vertices[2] - face_vertices[0];
        face_normal = e1.cross(e2).normalized();
    }
    
    // Compute data at each quadrature point on the face
    // For face integration, we need to transform the 2D face quadrature
    // to 3D space. The JxW factor becomes the area element on the face.
    
    for (int q = 0; q < n_qpoints_; ++q) {
        const auto& ip = fe_->quadrature_point(q);
        
        // For face integration, the Jacobian determinant is the area scaling
        // The quadrature weight needs to account for face area
        // This is simplified; proper implementation needs face Jacobian
        
        if (has_flag(flags_, UpdateFlags::UpdateJxW)) {
            // Compute face Jacobian (area element)
            // For triangles: J_face = 2 * area = |e1 x e2|
            // For quads: more complex
            Scalar face_detJ = 1.0;
            if (face_vertices.size() == 3) {
                // Triangle
                Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
                Tensor<1, 3> e2 = face_vertices[2] - face_vertices[0];
                face_detJ = 0.5 * e1.cross(e2).norm();
            } else if (face_vertices.size() == 4) {
                // Quadrilateral - approximate with average of two triangles
                Tensor<1, 3> e1 = face_vertices[1] - face_vertices[0];
                Tensor<1, 3> e2 = face_vertices[3] - face_vertices[0];
                Tensor<1, 3> e3 = face_vertices[2] - face_vertices[1];
                Tensor<1, 3> e4 = face_vertices[2] - face_vertices[3];
                face_detJ = 0.25 * (e1.cross(e2).norm() + e3.cross(e4).norm());
            }
            
            // For triangles, quadrature is in barycentric coordinates
            // The integration weight for a triangle with vertices v0,v1,v2:
            // integral = sum_q f(x_q) * w_q * Area
            // where Area = 0.5 * |(v1-v0) x (v2-v0)|
            // The quadrature weights are already scaled for reference triangle
            
            if (face_geom_type == GeometryType::Triangle) {
                // Triangle quadrature weights are already scaled for reference triangle
                // with area = 0.5. So JxW = weight * 2 * actual_area = weight * 4 * face_detJ
                qp_data_[q].JxW = ip.weight * 2.0 * face_detJ;
            } else if (face_geom_type == GeometryType::Quadrilateral) {
                // Quad quadrature weights are for [-1,1]^2 with area = 4
                // JxW = weight * actual_area / 4 * 4 = weight * actual_area
                qp_data_[q].JxW = ip.weight * face_detJ;
            } else {
                qp_data_[q].JxW = ip.weight * face_detJ;
            }
        }
        
        // Shape values (same as cell)
        if (has_flag(flags_, UpdateFlags::UpdateValues)) {
            for (int i = 0; i < dofs_per_cell_; ++i) {
                qp_data_[q].shape_values[i] = fe_->shape_value(i, q);
            }
        }
        
        // Physical gradients (for flux BC, etc.)
        if (has_flag(flags_, UpdateFlags::UpdateGradients)) {
            trans_.set_reference_point(ip.coord);
            for (int i = 0; i < dofs_per_cell_; ++i) {
                const Tensor<1, 3>& grad_ref = fe_->shape_gradient(i, q);
                qp_data_[q].shape_gradients[i] = trans_.transform_gradient(grad_ref);
            }
        }
        
        // Normal vector
        if (has_flag(flags_, UpdateFlags::UpdateNormals)) {
            qp_data_[q].normal = face_normal;
        }
        
        // Physical point on face
        if (has_flag(flags_, UpdateFlags::UpdateQuadraturePoints)) {
            // Interpolate from face vertices using shape functions
            // For simplicity, use cell transformation
            qp_data_[q].physical_point = trans_.transform(ip.coord);
        }
    }
}

void FEValues::get_function_values(const DynamicVector& global_solution,
                                   const std::vector<Index>& local_dofs,
                                   std::vector<Scalar>& qpoint_values) const {
    qpoint_values.resize(n_qpoints_, 0.0);
    
    // For scalar field
    if (n_components_ == 1) {
        for (int q = 0; q < n_qpoints_; ++q) {
            Scalar val = 0.0;
            for (int i = 0; i < dofs_per_cell_; ++i) {
                if (local_dofs[i] < global_solution.size()) {
                    val += qp_data_[q].shape_values[i] * global_solution[local_dofs[i]];
                }
            }
            qpoint_values[q] = val;
        }
    } else {
        // For vector field, take first component
        // (use get_vector_values for full vector)
        int stride = n_components_;
        for (int q = 0; q < n_qpoints_; ++q) {
            Scalar val = 0.0;
            for (int i = 0; i < dofs_per_cell_; ++i) {
                Index dof_idx = local_dofs[i] * stride;  // First component
                if (dof_idx < global_solution.size()) {
                    val += qp_data_[q].shape_values[i] * global_solution[dof_idx];
                }
            }
            qpoint_values[q] = val;
        }
    }
}

void FEValues::get_function_gradients(const DynamicVector& global_solution,
                                      const std::vector<Index>& local_dofs,
                                      std::vector<Tensor<1, 3>>& qpoint_grads) const {
    qpoint_grads.resize(n_qpoints_, Tensor<1, 3>::Zero());
    
    // For scalar field
    if (n_components_ == 1) {
        for (int q = 0; q < n_qpoints_; ++q) {
            Tensor<1, 3> grad = Tensor<1, 3>::Zero();
            for (int i = 0; i < dofs_per_cell_; ++i) {
                if (local_dofs[i] < global_solution.size()) {
                    grad += qp_data_[q].shape_gradients[i] * global_solution[local_dofs[i]];
                }
            }
            qpoint_grads[q] = grad;
        }
    } else {
        // For vector field, gradient of first component
        int stride = n_components_;
        for (int q = 0; q < n_qpoints_; ++q) {
            Tensor<1, 3> grad = Tensor<1, 3>::Zero();
            for (int i = 0; i < dofs_per_cell_; ++i) {
                Index dof_idx = local_dofs[i] * stride;  // First component
                if (dof_idx < global_solution.size()) {
                    grad += qp_data_[q].shape_gradients[i] * global_solution[dof_idx];
                }
            }
            qpoint_grads[q] = grad;
        }
    }
}

void FEValues::get_vector_values(const DynamicVector& global_solution,
                                 const std::vector<Index>& local_dofs,
                                 std::vector<Tensor<1, 3>>& qpoint_values) const {
    qpoint_values.resize(n_qpoints_, Tensor<1, 3>::Zero());
    
    // Assume 3D vector field
    int dim = std::min(n_components_, 3);
    
    for (int q = 0; q < n_qpoints_; ++q) {
        Tensor<1, 3> val = Tensor<1, 3>::Zero();
        for (int i = 0; i < dofs_per_cell_; ++i) {
            for (int c = 0; c < dim; ++c) {
                // DoF layout: interleaved (u0, v0, w0, u1, v1, w1, ...)
                // or block (u0, u1, ..., v0, v1, ..., w0, w1, ...)
                // Assuming interleaved for now
                Index dof_idx = local_dofs[i] + c;
                if (dof_idx < static_cast<Index>(global_solution.size())) {
                    val[c] += qp_data_[q].shape_values[i] * global_solution[dof_idx];
                }
            }
        }
        qpoint_values[q] = val;
    }
}

}  // namespace mpfem

/**
 * @file element_transformation.hpp
 * @brief Transformation from reference element to physical element
 */

#ifndef MPFEM_FEM_ELEMENT_TRANSFORMATION_HPP
#define MPFEM_FEM_ELEMENT_TRANSFORMATION_HPP

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include <Eigen/Dense>

namespace mpfem {

/**
 * @brief Transformation from reference element to physical element
 * 
 * Computes Jacobian matrix, its determinant, and inverse transpose
 * for mapping between reference and physical coordinates.
 */
class ElementTransformation {
public:
    ElementTransformation() = default;
    
    /// Reinitialize transformation for a given cell
    void reinit(const Mesh& mesh, Index cell_id);
    
    /// Reinitialize for a given face
    void reinit_face(const Mesh& mesh, Index face_id);
    
    /// Get Jacobian matrix at current quadrature point
    const Tensor<2, 3>& jacobian() const { return J_; }
    
    /// Get inverse of Jacobian matrix
    const Tensor<2, 3>& inverse_jacobian() const { return invJ_; }
    
    /// Get determinant of Jacobian
    Scalar det_jacobian() const { return detJ_; }
    
    /// Get quadrature weight multiplied by det(J)
    Scalar jxw(Scalar quad_weight) const { return detJ_ * quad_weight; }
    
    /// Transform reference gradient to physical gradient
    /// grad_phys = inv(J)^T * grad_ref
    Tensor<1, 3> transform_gradient(const Tensor<1, 3>& grad_ref) const;
    
    /// Transform physical gradient to reference gradient
    Tensor<1, 3> inverse_transform_gradient(const Tensor<1, 3>& grad_phys) const;
    
    /// Transform reference point to physical point
    Point<3> transform(const Point<3>& xi) const;
    
    /// Get cell vertices (for reference)
    const std::vector<Point<3>>& vertex_coords() const { return vertex_coords_; }
    
private:
    void compute_jacobian(const Point<3>& xi);
    
    Tensor<2, 3> J_;           ///< Jacobian matrix
    Tensor<2, 3> invJ_;        ///< Inverse Jacobian
    Scalar detJ_ = 1.0;        ///< Determinant of Jacobian
    
    std::vector<Point<3>> vertex_coords_;  ///< Physical vertex coordinates
    std::vector<Index> vertex_indices_;    ///< Vertex indices
    GeometryType geom_type_ = GeometryType::Invalid;
};

// Implementation

inline void ElementTransformation::reinit(const Mesh& mesh, Index cell_id) {
    // Get cell block and local cell index
    // For simplicity, find the block containing this cell
    const ElementBlock* block = nullptr;
    Index local_id = 0;
    
    for (const auto& b : mesh.cell_blocks()) {
        if (cell_id < static_cast<Index>(b.size())) {
            block = &b;
            local_id = cell_id;
            break;
        }
        cell_id -= static_cast<Index>(b.size());
    }
    
    if (!block) return;
    
    geom_type_ = to_geometry_type(block->type());
    
    // Get vertex coordinates
    int n_verts = block->nodes_per_element();
    vertex_coords_.resize(n_verts);
    vertex_indices_.resize(n_verts);
    
    auto verts = block->element_vertices(local_id);
    for (int i = 0; i < n_verts; ++i) {
        vertex_indices_[i] = verts[i];
        vertex_coords_[i] = mesh.vertex(vertex_indices_[i]);
    }
    
    // Compute Jacobian at element center (for initial setup)
    Point<3> xi_center;
    switch (geom_type_) {
        case GeometryType::Tetrahedron:
            xi_center = Point<3>(0.25, 0.25, 0.25);
            break;
        case GeometryType::Hexahedron:
            xi_center = Point<3>(0.0, 0.0, 0.0);
            break;
        default:
            xi_center = Point<3>(0.0, 0.0, 0.0);
    }
    
    compute_jacobian(xi_center);
}

inline void ElementTransformation::compute_jacobian(const Point<3>& xi) {
    // Compute Jacobian matrix using shape function gradients
    // J_ij = sum_k (dN_k/dxi_i * x_k_j)
    
    J_ = Tensor<2, 3>::Zero();
    
    if (geom_type_ == GeometryType::Tetrahedron && vertex_coords_.size() == 4) {
        // Linear tetrahedron: Jacobian is constant
        // N0 = 1-xi-eta-zeta, N1 = xi, N2 = eta, N3 = zeta
        // dN0 = (-1,-1,-1), dN1 = (1,0,0), dN2 = (0,1,0), dN3 = (0,0,1)
        
        for (int i = 0; i < 3; ++i) {
            J_(i, 0) = vertex_coords_[1][i] - vertex_coords_[0][i];
            J_(i, 1) = vertex_coords_[2][i] - vertex_coords_[0][i];
            J_(i, 2) = vertex_coords_[3][i] - vertex_coords_[0][i];
        }
    } else if (geom_type_ == GeometryType::Hexahedron && vertex_coords_.size() == 8) {
        // Linear hexahedron: Jacobian varies with xi
        const Scalar x = xi.x();
        const Scalar y = xi.y();
        const Scalar z = xi.z();
        
        // Shape function gradients at (x, y, z)
        // Node ordering: corners of [-1,1]^3
        // Node 0: (-1,-1,-1), Node 1: (1,-1,-1), ...
        Scalar dN[8][3];
        Scalar xi_vals[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
        Scalar eta_vals[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
        Scalar zeta_vals[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
        
        for (int n = 0; n < 8; ++n) {
            Scalar xi_n = xi_vals[n], eta_n = eta_vals[n], zeta_n = zeta_vals[n];
            dN[n][0] = 0.125 * xi_n * (1 + eta_n * y) * (1 + zeta_n * z);
            dN[n][1] = 0.125 * (1 + xi_n * x) * eta_n * (1 + zeta_n * z);
            dN[n][2] = 0.125 * (1 + xi_n * x) * (1 + eta_n * y) * zeta_n;
        }
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int n = 0; n < 8; ++n) {
                    J_(i, j) += dN[n][j] * vertex_coords_[n][i];
                }
            }
        }
    }
    
    // Compute determinant and inverse for 3x3 matrix
    detJ_ = J_.determinant();
    if (std::abs(detJ_) > 1e-15) {
        invJ_ = J_.inverse();
    } else {
        invJ_ = Tensor<2, 3>::Identity();
    }
}

inline Tensor<1, 3> ElementTransformation::transform_gradient(
    const Tensor<1, 3>& grad_ref) const {
    // grad_phys = inv(J)^T * grad_ref = (J^(-T)) * grad_ref
    return invJ_.transpose() * grad_ref;
}

inline Tensor<1, 3> ElementTransformation::inverse_transform_gradient(
    const Tensor<1, 3>& grad_phys) const {
    // grad_ref = J^T * grad_phys
    return J_.transpose() * grad_phys;
}

inline Point<3> ElementTransformation::transform(const Point<3>& xi) const {
    Point<3> x(0, 0, 0);
    
    if (geom_type_ == GeometryType::Tetrahedron && vertex_coords_.size() == 4) {
        const Scalar l0 = 1.0 - xi.x() - xi.y() - xi.z();
        x = l0 * vertex_coords_[0] + xi.x() * vertex_coords_[1] 
          + xi.y() * vertex_coords_[2] + xi.z() * vertex_coords_[3];
    } else if (geom_type_ == GeometryType::Hexahedron && vertex_coords_.size() == 8) {
        // Linear interpolation
        // This is simplified; full implementation needs shape functions
        for (size_t i = 0; i < vertex_coords_.size(); ++i) {
            // x += N_i(xi) * vertex_coords_[i];
        }
    }
    
    return x;
}

}  // namespace mpfem

#endif  // MPFEM_FEM_ELEMENT_TRANSFORMATION_HPP
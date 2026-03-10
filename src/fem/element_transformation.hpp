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
    const ElementBlock* block = nullptr;
    Index local_id = 0;
    Index remaining = cell_id;
    
    for (const auto& b : mesh.cell_blocks()) {
        if (remaining < static_cast<Index>(b.size())) {
            block = &b;
            local_id = remaining;
            break;
        }
        remaining -= static_cast<Index>(b.size());
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
    
    // Compute Jacobian at element center
    Point<3> xi_center;
    switch (geom_type_) {
        case GeometryType::Tetrahedron:
            xi_center = Point<3>(0.25, 0.25, 0.25);
            break;
        case GeometryType::Triangle:
            xi_center = Point<3>(1.0/3.0, 1.0/3.0, 0.0);
            break;
        case GeometryType::Hexahedron:
        case GeometryType::Pyramid:
            xi_center = Point<3>(0.0, 0.0, 0.0);
            break;
        case GeometryType::Wedge:
            xi_center = Point<3>(1.0/3.0, 1.0/3.0, 0.0);
            break;
        case GeometryType::Quadrilateral:
            xi_center = Point<3>(0.0, 0.0, 0.0);
            break;
        case GeometryType::Segment:
            xi_center = Point<3>(0.0, 0.0, 0.0);
            break;
        default:
            xi_center = Point<3>(0.0, 0.0, 0.0);
    }
    
    compute_jacobian(xi_center);
}

inline void ElementTransformation::compute_jacobian(const Point<3>& xi) {
    J_ = Tensor<2, 3>::Zero();
    
    const Scalar x = xi.x();
    const Scalar y = xi.y();
    const Scalar z = xi.z();
    
    switch (geom_type_) {
        case GeometryType::Tetrahedron: {
            // Linear tetrahedron: Jacobian is constant
            // N0 = 1-xi-eta-zeta, N1 = xi, N2 = eta, N3 = zeta
            if (vertex_coords_.size() >= 4) {
                for (int i = 0; i < 3; ++i) {
                    J_(i, 0) = vertex_coords_[1][i] - vertex_coords_[0][i];
                    J_(i, 1) = vertex_coords_[2][i] - vertex_coords_[0][i];
                    J_(i, 2) = vertex_coords_[3][i] - vertex_coords_[0][i];
                }
            }
            break;
        }
        
        case GeometryType::Hexahedron: {
            // Linear hexahedron: trilinear mapping
            if (vertex_coords_.size() >= 8) {
                // Node ordering: standard hex20 convention
                // 0:(-1,-1,-1), 1:(1,-1,-1), 2:(1,1,-1), 3:(-1,1,-1)
                // 4:(-1,-1,1), 5:(1,-1,1), 6:(1,1,1), 7:(-1,1,1)
                Scalar xi_n[8]  = {-1, 1, 1, -1, -1, 1, 1, -1};
                Scalar eta_n[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
                Scalar zeta_n[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
                
                for (int n = 0; n < 8; ++n) {
                    // dN/dxi = 0.125 * xi_n * (1 + eta_n*y) * (1 + zeta_n*z)
                    // dN/deta = 0.125 * (1 + xi_n*x) * eta_n * (1 + zeta_n*z)
                    // dN/dzeta = 0.125 * (1 + xi_n*x) * (1 + eta_n*y) * zeta_n
                    Scalar dN_dxi   = 0.125 * xi_n[n] * (1 + eta_n[n]*y) * (1 + zeta_n[n]*z);
                    Scalar dN_deta  = 0.125 * (1 + xi_n[n]*x) * eta_n[n] * (1 + zeta_n[n]*z);
                    Scalar dN_dzeta = 0.125 * (1 + xi_n[n]*x) * (1 + eta_n[n]*y) * zeta_n[n];
                    
                    for (int i = 0; i < 3; ++i) {
                        J_(i, 0) += dN_dxi * vertex_coords_[n][i];
                        J_(i, 1) += dN_deta * vertex_coords_[n][i];
                        J_(i, 2) += dN_dzeta * vertex_coords_[n][i];
                    }
                }
            }
            break;
        }
        
        case GeometryType::Wedge: {
            // Wedge = triangle x line
            // Reference coords: (xi, eta) on triangle [xi>=0, eta>=0, xi+eta<=1], zeta in [-1,1]
            if (vertex_coords_.size() >= 6) {
                // Node ordering for wedge:
                // Bottom triangle: 0:(0,0,-1), 1:(1,0,-1), 2:(0,1,-1)
                // Top triangle:    3:(0,0,1), 4:(1,0,1), 5:(0,1,1)
                Scalar l0 = 1.0 - x - y;  // Barycentric on triangle
                Scalar h0 = 0.5 * (1.0 - z);
                Scalar h1 = 0.5 * (1.0 + z);
                
                // Shape functions: N_i = l_j * h_k
                // Gradients: dN/dxi, dN/deta, dN/dzeta
                Scalar dN_dxi[6] = {-h0, h0, 0, -h1, h1, 0};
                Scalar dN_deta[6] = {-h0, 0, h0, -h1, 0, h1};
                Scalar dN_dzeta[6] = {-0.5*l0, -0.5*x, -0.5*y, 0.5*l0, 0.5*x, 0.5*y};
                
                for (int n = 0; n < 6; ++n) {
                    for (int i = 0; i < 3; ++i) {
                        J_(i, 0) += dN_dxi[n] * vertex_coords_[n][i];
                        J_(i, 1) += dN_deta[n] * vertex_coords_[n][i];
                        J_(i, 2) += dN_dzeta[n] * vertex_coords_[n][i];
                    }
                }
            }
            break;
        }
        
        case GeometryType::Pyramid: {
            // Pyramid with rational shape functions
            // Base: square on z=0, corners at (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)
            // Apex: (0,0,1)
            if (vertex_coords_.size() >= 5) {
                if (std::abs(z - 1.0) < 1e-14) {
                    // At apex: use limit values
                    for (int i = 0; i < 3; ++i) {
                        J_(i, 0) = 0.25 * (vertex_coords_[1][i] + vertex_coords_[2][i] 
                                          - vertex_coords_[0][i] - vertex_coords_[3][i]);
                        J_(i, 1) = 0.25 * (vertex_coords_[2][i] + vertex_coords_[3][i] 
                                          - vertex_coords_[0][i] - vertex_coords_[1][i]);
                        J_(i, 2) = 0.5 * (vertex_coords_[4][i] - 0.25 * (vertex_coords_[0][i] 
                                   + vertex_coords_[1][i] + vertex_coords_[2][i] + vertex_coords_[3][i]));
                    }
                } else {
                    Scalar omz = 1.0 - z;
                    Scalar px = 1.0 - x, mx = 1.0 + x;
                    Scalar py = 1.0 - y, my = 1.0 + y;
                    Scalar denom = 4.0 * omz;
                    
                    // Base node gradients
                    // N0 = (1-x)(1-y)/(4(1-z)) = px*py/denom
                    // dN0/dxi = -py/denom, dN0/deta = -px/denom, dN0/dzeta = px*py/(denom*(1-z))
                    Scalar dN_dxi[5] = {
                        -py/denom, my/denom, py/denom, -my/denom, 0
                    };
                    Scalar dN_deta[5] = {
                        -px/denom, -mx/denom, mx/denom, px/denom, 0
                    };
                    Scalar dN_dzeta[5] = {
                        px*py/(denom*omz), mx*py/(denom*omz), mx*my/(denom*omz), px*my/(denom*omz), 1.0
                    };
                    
                    for (int n = 0; n < 5; ++n) {
                        for (int i = 0; i < 3; ++i) {
                            J_(i, 0) += dN_dxi[n] * vertex_coords_[n][i];
                            J_(i, 1) += dN_deta[n] * vertex_coords_[n][i];
                            J_(i, 2) += dN_dzeta[n] * vertex_coords_[n][i];
                        }
                    }
                }
            }
            break;
        }
        
        case GeometryType::Triangle: {
            // 2D triangle in 3D space
            if (vertex_coords_.size() >= 3) {
                for (int i = 0; i < 3; ++i) {
                    J_(i, 0) = vertex_coords_[1][i] - vertex_coords_[0][i];
                    J_(i, 1) = vertex_coords_[2][i] - vertex_coords_[0][i];
                }
                // For 2D element, compute area via cross product
            }
            break;
        }
        
        case GeometryType::Quadrilateral: {
            // 2D quadrilateral in 3D space
            if (vertex_coords_.size() >= 4) {
                Scalar xi_n[4] = {-1, 1, 1, -1};
                Scalar eta_n[4] = {-1, -1, 1, 1};
                
                for (int n = 0; n < 4; ++n) {
                    Scalar dN_dxi = 0.25 * xi_n[n] * (1 + eta_n[n]*y);
                    Scalar dN_deta = 0.25 * (1 + xi_n[n]*x) * eta_n[n];
                    
                    for (int i = 0; i < 3; ++i) {
                        J_(i, 0) += dN_dxi * vertex_coords_[n][i];
                        J_(i, 1) += dN_deta * vertex_coords_[n][i];
                    }
                }
            }
            break;
        }
        
        case GeometryType::Segment: {
            // 1D segment in 3D space
            if (vertex_coords_.size() >= 2) {
                for (int i = 0; i < 3; ++i) {
                    J_(i, 0) = 0.5 * (vertex_coords_[1][i] - vertex_coords_[0][i]);
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    // Compute determinant and inverse
    if (geom_type_ == GeometryType::Triangle || geom_type_ == GeometryType::Quadrilateral) {
        // 2D element: compute area from cross product
        Tensor<1, 3> col0(J_(0,0), J_(1,0), J_(2,0));
        Tensor<1, 3> col1(J_(0,1), J_(1,1), J_(2,1));
        Tensor<1, 3> cross = col0.cross(col1);
        detJ_ = cross.norm();
    } else if (geom_type_ == GeometryType::Segment) {
        // 1D element: compute length
        detJ_ = std::sqrt(J_(0,0)*J_(0,0) + J_(1,0)*J_(1,0) + J_(2,0)*J_(2,0));
    } else {
        // 3D element: full determinant
        detJ_ = J_.determinant();
    }
    
    if (std::abs(detJ_) > 1e-15) {
        if (geom_type_ == GeometryType::Triangle || geom_type_ == GeometryType::Quadrilateral ||
            geom_type_ == GeometryType::Segment) {
            // For lower-dimensional elements, pseudo-inverse
            invJ_ = Tensor<2, 3>::Identity();
        } else {
            invJ_ = J_.inverse();
        }
    } else {
        invJ_ = Tensor<2, 3>::Identity();
    }
}

inline Tensor<1, 3> ElementTransformation::transform_gradient(
    const Tensor<1, 3>& grad_ref) const {
    return invJ_.transpose() * grad_ref;
}

inline Tensor<1, 3> ElementTransformation::inverse_transform_gradient(
    const Tensor<1, 3>& grad_phys) const {
    return J_.transpose() * grad_phys;
}

inline Point<3> ElementTransformation::transform(const Point<3>& xi) const {
    Point<3> x(0, 0, 0);
    const Scalar xcoord = xi.x();
    const Scalar ycoord = xi.y();
    const Scalar zcoord = xi.z();
    
    switch (geom_type_) {
        case GeometryType::Tetrahedron: {
            if (vertex_coords_.size() >= 4) {
                Scalar l0 = 1.0 - xcoord - ycoord - zcoord;
                x = l0 * vertex_coords_[0] + xcoord * vertex_coords_[1] 
                  + ycoord * vertex_coords_[2] + zcoord * vertex_coords_[3];
            }
            break;
        }
        
        case GeometryType::Hexahedron: {
            if (vertex_coords_.size() >= 8) {
                Scalar xi_n[8]  = {-1, 1, 1, -1, -1, 1, 1, -1};
                Scalar eta_n[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
                Scalar zeta_n[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
                
                for (int n = 0; n < 8; ++n) {
                    Scalar N = 0.125 * (1 + xi_n[n]*xcoord) * (1 + eta_n[n]*ycoord) * (1 + zeta_n[n]*zcoord);
                    x = x + N * vertex_coords_[n];
                }
            }
            break;
        }
        
        case GeometryType::Wedge: {
            if (vertex_coords_.size() >= 6) {
                Scalar l0 = 1.0 - xcoord - ycoord;
                Scalar h0 = 0.5 * (1.0 - zcoord);
                Scalar h1 = 0.5 * (1.0 + zcoord);
                
                Scalar N[6] = {l0*h0, xcoord*h0, ycoord*h0, l0*h1, xcoord*h1, ycoord*h1};
                for (int n = 0; n < 6; ++n) {
                    x = x + N[n] * vertex_coords_[n];
                }
            }
            break;
        }
        
        case GeometryType::Pyramid: {
            if (vertex_coords_.size() >= 5) {
                if (std::abs(zcoord - 1.0) < 1e-14) {
                    x = vertex_coords_[4];  // Apex
                } else {
                    Scalar omz = 1.0 - zcoord;
                    Scalar d = 1.0 / (4.0 * omz);
                    
                    Scalar N[5] = {
                        (1.0 - xcoord) * (1.0 - ycoord) * omz * d,
                        (1.0 + xcoord) * (1.0 - ycoord) * omz * d,
                        (1.0 + xcoord) * (1.0 + ycoord) * omz * d,
                        (1.0 - xcoord) * (1.0 + ycoord) * omz * d,
                        zcoord
                    };
                    for (int n = 0; n < 5; ++n) {
                        x = x + N[n] * vertex_coords_[n];
                    }
                }
            }
            break;
        }
        
        case GeometryType::Triangle: {
            if (vertex_coords_.size() >= 3) {
                Scalar l0 = 1.0 - xcoord - ycoord;
                x = l0 * vertex_coords_[0] + xcoord * vertex_coords_[1] + ycoord * vertex_coords_[2];
            }
            break;
        }
        
        case GeometryType::Quadrilateral: {
            if (vertex_coords_.size() >= 4) {
                Scalar xi_n[4] = {-1, 1, 1, -1};
                Scalar eta_n[4] = {-1, -1, 1, 1};
                
                for (int n = 0; n < 4; ++n) {
                    Scalar N = 0.25 * (1 + xi_n[n]*xcoord) * (1 + eta_n[n]*ycoord);
                    x = x + N * vertex_coords_[n];
                }
            }
            break;
        }
        
        case GeometryType::Segment: {
            if (vertex_coords_.size() >= 2) {
                Scalar N0 = 0.5 * (1.0 - xcoord);
                Scalar N1 = 0.5 * (1.0 + xcoord);
                x = N0 * vertex_coords_[0] + N1 * vertex_coords_[1];
            }
            break;
        }
        
        default:
            break;
    }
    
    return x;
}

inline void ElementTransformation::reinit_face(const Mesh& mesh, Index face_id) {
    const ElementBlock* block = nullptr;
    Index local_id = 0;
    Index remaining = face_id;
    
    for (const auto& b : mesh.face_blocks()) {
        if (remaining < static_cast<Index>(b.size())) {
            block = &b;
            local_id = remaining;
            break;
        }
        remaining -= static_cast<Index>(b.size());
    }
    
    if (!block) return;
    
    geom_type_ = to_geometry_type(block->type());
    
    int n_verts = block->nodes_per_element();
    vertex_coords_.resize(n_verts);
    vertex_indices_.resize(n_verts);
    
    auto verts = block->element_vertices(local_id);
    for (int i = 0; i < n_verts; ++i) {
        vertex_indices_[i] = verts[i];
        vertex_coords_[i] = mesh.vertex(vertex_indices_[i]);
    }
    
    compute_jacobian(Point<3>(1.0/3.0, 1.0/3.0, 0.0));
}

}  // namespace mpfem

#endif  // MPFEM_FEM_ELEMENT_TRANSFORMATION_HPP

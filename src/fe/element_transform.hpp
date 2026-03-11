#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/shape_function.hpp"
#include "fe/quadrature.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <functional>

namespace mpfem {

/**
 * @brief Element transformation from reference to physical coordinates.
 * 
 * Provides mapping between reference element and physical element,
 * including Jacobian computation and coordinate transformation.
 * 
 * Design inspired by MFEM's ElementTransformation class.
 */
class ElementTransform {
public:
    /// Default constructor
    ElementTransform() = default;
    
    /// Construct with mesh and element index
    ElementTransform(const Mesh* mesh, Index elemIdx, bool isBoundary = false);
    
    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    
    /// Set the mesh
    void setMesh(const Mesh* mesh) { mesh_ = mesh; }
    
    /// Set the element index
    void setElement(Index elemIdx, bool isBoundary = false);
    
    /// Set integration point (invalidates cached Jacobian data)
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Real* xi);
    
    /// Reset evaluation state (force recompute on next access)
    void reset() { evalState_ = 0; }
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension
    int dim() const { return dim_; }
    
    /// Get element index
    Index elementIndex() const { return elemIdx_; }
    
    /// Is this a boundary element?
    bool isBoundary() const { return isBoundary_; }
    
    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }
    
    // -------------------------------------------------------------------------
    // Transformation
    // -------------------------------------------------------------------------
    
    /// Transform reference coordinates to physical coordinates
    void transform(const Real* xi, Real* x) const;
    void transform(const Real* xi, Vector3& x) const;
    
    /// Transform integration point to physical coordinates
    void transform(const IntegrationPoint& ip, Vector3& x) const {
        transform(&ip.xi, x);
    }
    
    // -------------------------------------------------------------------------
    // Jacobian and related quantities
    // -------------------------------------------------------------------------
    
    /// Get the Jacobian matrix (dim x dim)
    /// J_ij = dx_i / dxi_j
    const Matrix& jacobian() const;
    
    /// Get the inverse Jacobian matrix
    const Matrix& invJacobian() const;
    
    /// Get the Jacobian determinant
    Real detJ() const;
    
    /// Get the weight = |det(J)|
    Real weight() const;
    
    /// Get the adjugate of the Jacobian (for gradient transformation)
    const Matrix& adjJacobian() const;
    
    // -------------------------------------------------------------------------
    // Gradient transformation
    // -------------------------------------------------------------------------
    
    /**
     * @brief Transform gradient from reference to physical coordinates.
     * 
     * Given gradient in reference coordinates: grad_xi(phi)
     * Returns gradient in physical coordinates: grad_x(phi)
     * 
     * grad_x(phi) = J^{-T} * grad_xi(phi)
     */
    void transformGradient(const Real* refGrad, Real* physGrad) const;
    void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;
    
    // -------------------------------------------------------------------------
    // Normal vector (for boundary elements)
    // -------------------------------------------------------------------------
    
    /// Get outward normal vector at current integration point (2D/3D)
    Vector3 normal() const;
    
    // -------------------------------------------------------------------------
    // Element vertices (convenience)
    // -------------------------------------------------------------------------
    
    /// Get vertex coordinates
    const std::vector<Vector3>& vertices() const { return vertices_; }
    
    /// Get number of vertices
    int numVertices() const { return static_cast<int>(vertices_.size()); }
    
    /// Get the currently set integration point
    const IntegrationPoint& integrationPoint() const { return ip_; }
    
private:
    // -------------------------------------------------------------------------
    // Evaluation state management (MFEM-style)
    // -------------------------------------------------------------------------
    enum EvalMask {
        JACOBIAN_MASK = 1,
        WEIGHT_MASK   = 2,
        ADJUGATE_MASK = 4,
        INVERSE_MASK  = 8
    };
    
    void computeGeometryInfo();
    void evalJacobian() const;
    void evalWeight() const;
    void evalAdjugate() const;
    void evalInverse() const;
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    const Mesh* mesh_ = nullptr;
    Index elemIdx_ = 0;
    bool isBoundary_ = false;
    
    Geometry geometry_ = Geometry::Invalid;
    int dim_ = 0;
    std::vector<Vector3> vertices_;
    std::vector<Index> vertexIndices_;
    
    IntegrationPoint ip_;
    
    // Mutable for lazy evaluation
    mutable Matrix jacobian_;
    mutable Matrix invJacobian_;
    mutable Matrix adjJacobian_;
    mutable Real detJ_ = 0.0;
    mutable Real weight_ = 0.0;
    mutable int evalState_ = 0;
    
    // Shape function storage for Jacobian computation
    mutable ShapeValues shapeValues_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline ElementTransform::ElementTransform(const Mesh* mesh, Index elemIdx, bool isBoundary)
    : mesh_(mesh), elemIdx_(elemIdx), isBoundary_(isBoundary) {
    computeGeometryInfo();
}

inline void ElementTransform::setElement(Index elemIdx, bool isBoundary) {
    elemIdx_ = elemIdx;
    isBoundary_ = isBoundary;
    evalState_ = 0;
    computeGeometryInfo();
}

inline void ElementTransform::setIntegrationPoint(const IntegrationPoint& ip) {
    ip_ = ip;
    evalState_ = 0;
}

inline void ElementTransform::setIntegrationPoint(const Real* xi) {
    ip_.xi = xi[0];
    if (dim_ > 1) ip_.eta = xi[1];
    if (dim_ > 2) ip_.zeta = xi[2];
    evalState_ = 0;
}

inline void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (isBoundary_) {
        if (elemIdx_ < mesh_->numBdrElements()) {
            elem = &mesh_->bdrElement(elemIdx_);
        }
    } else {
        if (elemIdx_ < mesh_->numElements()) {
            elem = &mesh_->element(elemIdx_);
        }
    }
    
    if (!elem) return;
    
    geometry_ = elem->geometry();
    dim_ = geom::dim(geometry_);
    
    // Get vertex coordinates
    vertexIndices_ = elem->vertices();
    vertices_.resize(vertexIndices_.size());
    for (size_t i = 0; i < vertexIndices_.size(); ++i) {
        vertices_[i] = mesh_->vertex(vertexIndices_[i]).toVector();
    }
    
    // Pre-allocate matrices
    jacobian_.setZero(dim_, dim_);
    invJacobian_.setZero(dim_, dim_);
    adjJacobian_.setZero(dim_, dim_);
}

// =============================================================================
// Jacobian evaluation
// =============================================================================

inline void ElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    // For linear elements, Jacobian is constant
    // J_ij = dx_i / dxi_j
    
    switch (geometry_) {
        case Geometry::Segment: {
            // dphi/dxi = (-0.5, 0.5)
            // J = 0.5 * (x1 - x0)
            jacobian_(0, 0) = 0.5 * (vertices_[1][0] - vertices_[0][0]);
            break;
        }
        
        case Geometry::Triangle: {
            // Shape function gradients in reference coords (constant for linear)
            // phi0 = 1 - xi - eta, phi1 = xi, phi2 = eta
            // grad(phi0) = (-1, -1), grad(phi1) = (1, 0), grad(phi2) = (0, 1)
            // J = [x1-x0, x2-x0; y1-y0, y2-y0]
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Square: {
            // Bilinear mapping: phi = 0.25 * (1 +/- xi) * (1 +/- eta)
            Real xi = ip_.xi, eta = ip_.eta;
            
            // dphi/dxi
            Real dphi0_dxi = -0.25 * (1.0 - eta);
            Real dphi1_dxi =  0.25 * (1.0 - eta);
            Real dphi2_dxi =  0.25 * (1.0 + eta);
            Real dphi3_dxi = -0.25 * (1.0 + eta);
            
            // dphi/deta
            Real dphi0_deta = -0.25 * (1.0 - xi);
            Real dphi1_deta = -0.25 * (1.0 + xi);
            Real dphi2_deta =  0.25 * (1.0 + xi);
            Real dphi3_deta =  0.25 * (1.0 - xi);
            
            for (int i = 0; i < 2; ++i) {
                jacobian_(i, 0) = dphi0_dxi * vertices_[0][i] + dphi1_dxi * vertices_[1][i] +
                                  dphi2_dxi * vertices_[2][i] + dphi3_dxi * vertices_[3][i];
                jacobian_(i, 1) = dphi0_deta * vertices_[0][i] + dphi1_deta * vertices_[1][i] +
                                  dphi2_deta * vertices_[2][i] + dphi3_deta * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Tetrahedron: {
            // Linear tetrahedron: constant Jacobian
            // phi0 = 1 - xi - eta - zeta, phi1 = xi, phi2 = eta, phi3 = zeta
            // J_ij = x_{j+1,i} - x_{0,i} for j = 0,1,2
            for (int i = 0; i < 3; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
                jacobian_(i, 2) = vertices_[3][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Cube: {
            // Trilinear mapping for hexahedron
            Real xi = ip_.xi, eta = ip_.eta, zeta = ip_.zeta;
            
            // Derivatives of shape functions
            // phi_k = 0.125 * (1 +/- xi) * (1 +/- eta) * (1 +/- zeta)
            Real dphi[8][3];  // dphi/dxi, dphi/deta, dphi/dzeta
            
            // Vertex ordering: (following geometry.hpp face_table::Cube)
            // 0: (-1,-1,-1), 1: (1,-1,-1), 2: (1,1,-1), 3: (-1,1,-1)
            // 4: (-1,-1,1),  5: (1,-1,1),  6: (1,1,1),  7: (-1,1,1)
            
            Real t1 = 0.125 * (1.0 - eta) * (1.0 - zeta);
            Real t2 = 0.125 * (1.0 + eta) * (1.0 - zeta);
            Real t3 = 0.125 * (1.0 - eta) * (1.0 + zeta);
            Real t4 = 0.125 * (1.0 + eta) * (1.0 + zeta);
            
            Real s1 = 0.125 * (1.0 - xi) * (1.0 - zeta);
            Real s2 = 0.125 * (1.0 + xi) * (1.0 - zeta);
            Real s3 = 0.125 * (1.0 - xi) * (1.0 + zeta);
            Real s4 = 0.125 * (1.0 + xi) * (1.0 + zeta);
            
            Real r1 = 0.125 * (1.0 - xi) * (1.0 - eta);
            Real r2 = 0.125 * (1.0 + xi) * (1.0 - eta);
            Real r3 = 0.125 * (1.0 + xi) * (1.0 + eta);
            Real r4 = 0.125 * (1.0 - xi) * (1.0 + eta);
            
            // dphi/dxi
            dphi[0][0] = -t1; dphi[1][0] = t1;
            dphi[2][0] = t2;  dphi[3][0] = -t2;
            dphi[4][0] = -t3; dphi[5][0] = t3;
            dphi[6][0] = t4;  dphi[7][0] = -t4;
            
            // dphi/deta
            dphi[0][1] = -s1; dphi[1][1] = -s2;
            dphi[2][1] = s2;  dphi[3][1] = s1;
            dphi[4][1] = -s3; dphi[5][1] = -s4;
            dphi[6][1] = s4;  dphi[7][1] = s3;
            
            // dphi/dzeta
            dphi[0][2] = -r1; dphi[1][2] = -r2;
            dphi[2][2] = -r3; dphi[3][2] = -r4;
            dphi[4][2] = r1;  dphi[5][2] = r2;
            dphi[6][2] = r3;  dphi[7][2] = r4;
            
            // J = sum_k (x_k * grad(phi_k))
            for (int i = 0; i < 3; ++i) {
                jacobian_(i, 0) = 0;
                jacobian_(i, 1) = 0;
                jacobian_(i, 2) = 0;
                for (int k = 0; k < 8; ++k) {
                    jacobian_(i, 0) += vertices_[k][i] * dphi[k][0];
                    jacobian_(i, 1) += vertices_[k][i] * dphi[k][1];
                    jacobian_(i, 2) += vertices_[k][i] * dphi[k][2];
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    evalState_ |= JACOBIAN_MASK;
}

inline void ElementTransform::evalWeight() const {
    if (evalState_ & WEIGHT_MASK) return;
    
    // Ensure Jacobian is computed
    evalJacobian();
    
    // Compute determinant based on dimension
    if (dim_ == 1) {
        detJ_ = jacobian_(0, 0);
    } else if (dim_ == 2) {
        detJ_ = jacobian_(0, 0) * jacobian_(1, 1) - jacobian_(0, 1) * jacobian_(1, 0);
    } else if (dim_ == 3) {
        detJ_ = jacobian_(0, 0) * (jacobian_(1, 1) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 1))
              - jacobian_(0, 1) * (jacobian_(1, 0) * jacobian_(2, 2) - jacobian_(1, 2) * jacobian_(2, 0))
              + jacobian_(0, 2) * (jacobian_(1, 0) * jacobian_(2, 1) - jacobian_(1, 1) * jacobian_(2, 0));
    }
    
    weight_ = std::abs(detJ_);
    evalState_ |= WEIGHT_MASK;
}

inline void ElementTransform::evalAdjugate() const {
    if (evalState_ & ADJUGATE_MASK) return;
    
    // Ensure Jacobian and weight are computed
    evalWeight();
    
    if (dim_ == 1) {
        adjJacobian_(0, 0) = 1.0;
    } else if (dim_ == 2) {
        adjJacobian_(0, 0) = jacobian_(1, 1);
        adjJacobian_(0, 1) = -jacobian_(0, 1);
        adjJacobian_(1, 0) = -jacobian_(1, 0);
        adjJacobian_(1, 1) = jacobian_(0, 0);
    } else if (dim_ == 3) {
        // adj(J) = det(J) * J^{-1}, computed after inverse
        evalInverse();
        adjJacobian_ = detJ_ * invJacobian_;
    }
    
    evalState_ |= ADJUGATE_MASK;
}

inline void ElementTransform::evalInverse() const {
    if (evalState_ & INVERSE_MASK) return;
    
    // Ensure Jacobian is computed
    evalJacobian();
    
    if (dim_ == 1) {
        invJacobian_(0, 0) = 1.0 / jacobian_(0, 0);
    } else if (dim_ == 2) {
        evalWeight();
        Real invDet = 1.0 / detJ_;
        invJacobian_(0, 0) = jacobian_(1, 1) * invDet;
        invJacobian_(0, 1) = -jacobian_(0, 1) * invDet;
        invJacobian_(1, 0) = -jacobian_(1, 0) * invDet;
        invJacobian_(1, 1) = jacobian_(0, 0) * invDet;
    } else if (dim_ == 3) {
        invJacobian_ = jacobian_.inverse();
    }
    
    evalState_ |= INVERSE_MASK;
}

// =============================================================================
// Accessor implementations
// =============================================================================

inline const Matrix& ElementTransform::jacobian() const {
    evalJacobian();
    return jacobian_;
}

inline const Matrix& ElementTransform::invJacobian() const {
    evalInverse();
    return invJacobian_;
}

inline Real ElementTransform::detJ() const {
    evalWeight();
    return detJ_;
}

inline Real ElementTransform::weight() const {
    evalWeight();
    return weight_;
}

inline const Matrix& ElementTransform::adjJacobian() const {
    evalAdjugate();
    return adjJacobian_;
}

// =============================================================================
// Transform implementations
// =============================================================================

inline void ElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    switch (geometry_) {
        case Geometry::Segment: {
            // Reference: [-1, 1]
            // phi0 = 0.5*(1-xi), phi1 = 0.5*(1+xi)
            Real phi0 = 0.5 * (1.0 - xi[0]);
            Real phi1 = 0.5 * (1.0 + xi[0]);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi0 * vertices_[0][i] + phi1 * vertices_[1][i];
            }
            break;
        }
        
        case Geometry::Triangle: {
            // Reference: (0,0), (1,0), (0,1)
            // Barycentric: (1-xi-eta, xi, eta)
            Real l1 = 1.0 - xi[0] - xi[1];
            Real l2 = xi[0];
            Real l3 = xi[1];
            for (int i = 0; i < 3; ++i) {
                x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] + l3 * vertices_[2][i];
            }
            break;
        }
        
        case Geometry::Square: {
            // Reference: [-1,1] x [-1,1]
            Real xi1 = xi[0], xi2 = xi[1];
            Real phi[4];
            phi[0] = 0.25 * (1.0 - xi1) * (1.0 - xi2);
            phi[1] = 0.25 * (1.0 + xi1) * (1.0 - xi2);
            phi[2] = 0.25 * (1.0 + xi1) * (1.0 + xi2);
            phi[3] = 0.25 * (1.0 - xi1) * (1.0 + xi2);
            for (int i = 0; i < 3; ++i) {
                x[i] = phi[0] * vertices_[0][i] + phi[1] * vertices_[1][i] +
                       phi[2] * vertices_[2][i] + phi[3] * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Tetrahedron: {
            // Reference: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            // Barycentric: (1-xi-eta-zeta, xi, eta, zeta)
            Real l1 = 1.0 - xi[0] - xi[1] - xi[2];
            Real l2 = xi[0];
            Real l3 = xi[1];
            Real l4 = xi[2];
            for (int i = 0; i < 3; ++i) {
                x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] +
                       l3 * vertices_[2][i] + l4 * vertices_[3][i];
            }
            break;
        }
        
        case Geometry::Cube: {
            // Reference: [-1,1]^3
            Real xi1 = xi[0], xi2 = xi[1], xi3 = xi[2];
            Real phi[8];
            phi[0] = 0.125 * (1.0 - xi1) * (1.0 - xi2) * (1.0 - xi3);
            phi[1] = 0.125 * (1.0 + xi1) * (1.0 - xi2) * (1.0 - xi3);
            phi[2] = 0.125 * (1.0 + xi1) * (1.0 + xi2) * (1.0 - xi3);
            phi[3] = 0.125 * (1.0 - xi1) * (1.0 + xi2) * (1.0 - xi3);
            phi[4] = 0.125 * (1.0 - xi1) * (1.0 - xi2) * (1.0 + xi3);
            phi[5] = 0.125 * (1.0 + xi1) * (1.0 - xi2) * (1.0 + xi3);
            phi[6] = 0.125 * (1.0 + xi1) * (1.0 + xi2) * (1.0 + xi3);
            phi[7] = 0.125 * (1.0 - xi1) * (1.0 + xi2) * (1.0 + xi3);
            for (int i = 0; i < 3; ++i) {
                x[i] = 0;
                for (int j = 0; j < 8; ++j) {
                    x[i] += phi[j] * vertices_[j][i];
                }
            }
            break;
        }
        
        default:
            break;
    }
}

inline void ElementTransform::transform(const Real* xi, Vector3& x) const {
    Real coords[3];
    transform(xi, coords);
    x = Vector3(coords[0], coords[1], coords[2]);
}

inline void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    // grad_x(phi) = J^{-T} * grad_xi(phi)
    // = (J^{-1})^T * refGrad
    // = invJacobian^T * refGrad
    
    const Matrix& invJ = invJacobian();
    
    for (int i = 0; i < dim_; ++i) {
        physGrad[i] = 0.0;
        for (int j = 0; j < dim_; ++j) {
            physGrad[i] += invJ(j, i) * refGrad[j];
        }
    }
}

inline void ElementTransform::transformGradient(const Vector3& refGrad, Vector3& physGrad) const {
    Real rg[3] = {refGrad.x(), refGrad.y(), refGrad.z()};
    Real pg[3];
    transformGradient(rg, pg);
    physGrad = Vector3(pg[0], pg[1], pg[2]);
}

inline Vector3 ElementTransform::normal() const {
    // Normal vector for surface elements
    // n = (J[:,0] x J[:,1]) / |J[:,0] x J[:,1]| for 2D surface in 3D
    // n = (J[:,0]) / |J[:,0]| rotated by 90 deg for 1D curve in 2D
    
    const Matrix& J = jacobian();
    Vector3 n(0.0, 0.0, 0.0);
    
    if (dim_ == 2 && geometry_ == Geometry::Triangle) {
        // Triangle in 3D space
        // n = (dF/dxi) x (dF/deta) / |...|
        Vector3 d1(J(0, 0), J(1, 0), J(2, 0));
        Vector3 d2(J(0, 1), J(1, 1), J(2, 1));
        n = d1.cross(d2);
        n.normalize();
    } else if (dim_ == 2 && geometry_ == Geometry::Square) {
        // Quadrilateral in 3D space
        Vector3 d1(J(0, 0), J(1, 0), J(2, 0));
        Vector3 d2(J(0, 1), J(1, 1), J(2, 1));
        n = d1.cross(d2);
        n.normalize();
    } else if (dim_ == 1 && geometry_ == Geometry::Segment) {
        // Segment in 2D space - normal is tangent rotated by 90 degrees
        // Assuming the segment lies in the xy-plane
        Real tx = J(0, 0);
        Real ty = J(1, 0);
        n = Vector3(-ty, tx, 0.0);
        n.normalize();
    }
    
    return n;
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_TRANSFORM_HPP
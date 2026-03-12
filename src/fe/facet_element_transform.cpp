#include "fe/facet_element_transform.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// FacetElementTransform Implementation
// =============================================================================

Index FacetElementTransform::boundaryAttribute() const {
    if (!mesh_) return 0;
    
    if (bdrElemIdx_ < mesh_->numBdrElements()) {
        return mesh_->bdrElement(bdrElemIdx_).attribute();
    }
    return 0;
}

void FacetElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    if (bdrElemIdx_ >= mesh_->numBdrElements()) return;
    
    const Element& elem = mesh_->bdrElement(bdrElemIdx_);
    geometry_ = elem.geometry();
    spaceDim_ = mesh_->dim();
    dim_ = geom::dim(geometry_);  // Dimension of the boundary element
    
    // Get vertex coordinates
    vertexIndices_ = elem.vertices();
    vertices_.resize(vertexIndices_.size());
    for (size_t i = 0; i < vertexIndices_.size(); ++i) {
        vertices_[i] = mesh_->vertex(vertexIndices_[i]).toVector();
    }
    
    // Pre-allocate Jacobian matrix (spaceDim_ x dim_)
    jacobian_.setZero(spaceDim_, dim_);
}

// =============================================================================
// Jacobian evaluation
// =============================================================================

void FacetElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    switch (geometry_) {
        case Geometry::Segment: {
            // Segment in 2D or 3D space
            // Reference: [-1, 1]
            // dphi/dxi = (-0.5, 0.5)
            // J = 0.5 * (x1 - x0)  -- tangent vector
            for (int i = 0; i < spaceDim_; ++i) {
                jacobian_(i, 0) = 0.5 * (vertices_[1][i] - vertices_[0][i]);
            }
            break;
        }
        
        case Geometry::Triangle: {
            // Triangle in 3D space
            // Reference: (0,0), (1,0), (0,1)
            // Barycentric: (1-xi-eta, xi, eta)
            // grad_xi(phi) = (-1, 1, 0), grad_eta(phi) = (-1, 0, 1)
            // J = [x1-x0, x2-x0; y1-y0, y2-y0; z1-z0, z2-z0]
            for (int i = 0; i < spaceDim_; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Square: {
            // Quadrilateral in 3D space
            // Reference: [-1,1] x [-1,1]
            Real xi = ip_.xi, eta = ip_.eta;
            
            // Bilinear shape function derivatives
            Real dphi0_dxi = -0.25 * (1.0 - eta);
            Real dphi1_dxi =  0.25 * (1.0 - eta);
            Real dphi2_dxi =  0.25 * (1.0 + eta);
            Real dphi3_dxi = -0.25 * (1.0 + eta);
            
            Real dphi0_deta = -0.25 * (1.0 - xi);
            Real dphi1_deta = -0.25 * (1.0 + xi);
            Real dphi2_deta =  0.25 * (1.0 + xi);
            Real dphi3_deta =  0.25 * (1.0 - xi);
            
            for (int i = 0; i < spaceDim_; ++i) {
                jacobian_(i, 0) = dphi0_dxi * vertices_[0][i] + dphi1_dxi * vertices_[1][i] +
                                  dphi2_dxi * vertices_[2][i] + dphi3_dxi * vertices_[3][i];
                jacobian_(i, 1) = dphi0_deta * vertices_[0][i] + dphi1_deta * vertices_[1][i] +
                                  dphi2_deta * vertices_[2][i] + dphi3_deta * vertices_[3][i];
            }
            break;
        }
        
        default:
            break;
    }
    
    evalState_ |= JACOBIAN_MASK;
}

void FacetElementTransform::evalWeight() const {
    if (evalState_ & WEIGHT_MASK) return;
    
    // Ensure Jacobian is computed
    evalJacobian();
    
    // For surface elements, the "det(J)" is the area scaling factor
    // This equals the magnitude of the cross product of Jacobian columns
    
    if (dim_ == 1) {
        // 1D element (segment) in 2D or 3D space
        // detJ = length of tangent vector = |J[:,0]|
        Real sum = 0.0;
        for (int i = 0; i < spaceDim_; ++i) {
            sum += jacobian_(i, 0) * jacobian_(i, 0);
        }
        detJ_ = std::sqrt(sum);
    } else if (dim_ == 2) {
        // 2D element (triangle or quad) in 3D space
        // detJ = |J[:,0] x J[:,1]|
        Vector3 d1(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
        Vector3 d2(jacobian_(0, 1), jacobian_(1, 1), jacobian_(2, 1));
        Vector3 cross = d1.cross(d2);
        detJ_ = cross.norm();
    }
    
    weight_ = detJ_;
    evalState_ |= WEIGHT_MASK;
}

// =============================================================================
// Transform implementations
// =============================================================================

void FacetElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    switch (geometry_) {
        case Geometry::Segment: {
            // Reference: [-1, 1]
            Real phi0 = 0.5 * (1.0 - xi[0]);
            Real phi1 = 0.5 * (1.0 + xi[0]);
            for (int i = 0; i < spaceDim_; ++i) {
                x[i] = phi0 * vertices_[0][i] + phi1 * vertices_[1][i];
            }
            break;
        }
        
        case Geometry::Triangle: {
            // Reference: (0,0), (1,0), (0,1)
            Real l1 = 1.0 - xi[0] - xi[1];
            Real l2 = xi[0];
            Real l3 = xi[1];
            for (int i = 0; i < spaceDim_; ++i) {
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
            for (int i = 0; i < spaceDim_; ++i) {
                x[i] = phi[0] * vertices_[0][i] + phi[1] * vertices_[1][i] +
                       phi[2] * vertices_[2][i] + phi[3] * vertices_[3][i];
            }
            break;
        }
        
        default:
            break;
    }
}

Vector3 FacetElementTransform::normal() const {
    const Matrix& J = jacobian();
    Vector3 n(0.0, 0.0, 0.0);
    
    if (dim_ == 1) {
        // Segment in 2D or 3D
        // Normal is tangent rotated 90 degrees (for 2D) or undefined (for 3D)
        if (spaceDim_ == 2) {
            // Segment in xy-plane: tangent = (J(0,0), J(1,0))
            // Normal = (-J(1,0), J(0,0)) / |...|
            Real tx = J(0, 0);
            Real ty = J(1, 0);
            Real len = std::sqrt(tx * tx + ty * ty);
            if (len > 0) {
                n = Vector3(-ty / len, tx / len, 0.0);
            }
        } else if (spaceDim_ == 3) {
            // For segment in 3D, we need additional information to define normal
            // This is typically provided by the adjacent element's face normal
            // For now, compute a perpendicular direction
            Vector3 tangent(J(0, 0), J(1, 0), J(2, 0));
            tangent.normalize();
            
            // Find a vector not parallel to tangent
            Vector3 other(1.0, 0.0, 0.0);
            if (std::abs(tangent.dot(other)) > 0.9) {
                other = Vector3(0.0, 1.0, 0.0);
            }
            
            // Normal is perpendicular to tangent
            n = tangent.cross(other);
            n.normalize();
        }
    } else if (dim_ == 2) {
        // Triangle or quad in 3D
        // n = (dF/dxi) x (dF/deta) / |...|
        Vector3 d1(J(0, 0), J(1, 0), J(2, 0));
        Vector3 d2(J(0, 1), J(1, 1), J(2, 1));
        n = d1.cross(d2);
        n.normalize();
    }
    
    return n;
}

}  // namespace mpfem

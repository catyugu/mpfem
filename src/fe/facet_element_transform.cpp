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
    
    // Get geometric order and create shape function
    geomOrder_ = elem.order();
    createGeomShapeFunction();
}

void FacetElementTransform::createGeomShapeFunction() {
    switch (geometry_) {
        case Geometry::Segment:
            geomShapeFunc_ = std::make_unique<H1SegmentShape>(geomOrder_);
            break;
        case Geometry::Triangle:
            geomShapeFunc_ = std::make_unique<H1TriangleShape>(geomOrder_);
            break;
        case Geometry::Square:
            geomShapeFunc_ = std::make_unique<H1SquareShape>(geomOrder_);
            break;
        default:
            geomShapeFunc_ = nullptr;
            break;
    }
}

// =============================================================================
// Transform implementations
// =============================================================================

void FacetElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    if (!geomShapeFunc_) {
        // Fallback to linear interpolation for unsupported geometry
        switch (geometry_) {
            case Geometry::Segment: {
                Real phi0 = 0.5 * (1.0 - xi[0]);
                Real phi1 = 0.5 * (1.0 + xi[0]);
                for (int i = 0; i < spaceDim_; ++i) {
                    x[i] = phi0 * vertices_[0][i] + phi1 * vertices_[1][i];
                }
                break;
            }
            case Geometry::Triangle: {
                Real l1 = 1.0 - xi[0] - xi[1];
                Real l2 = xi[0];
                Real l3 = xi[1];
                for (int i = 0; i < spaceDim_; ++i) {
                    x[i] = l1 * vertices_[0][i] + l2 * vertices_[1][i] + l3 * vertices_[2][i];
                }
                break;
            }
            case Geometry::Square: {
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
        return;
    }
    
    // Use geometric shape functions for coordinate interpolation
    auto sv = geomShapeFunc_->evalValues(xi);
    
    for (int i = 0; i < spaceDim_; ++i) {
        x[i] = 0.0;
    }
    
    for (int j = 0; j < geomShapeFunc_->numDofs(); ++j) {
        for (int i = 0; i < spaceDim_; ++i) {
            x[i] += sv[j] * vertices_[j][i];
        }
    }
}

// =============================================================================
// Jacobian evaluation
// =============================================================================

void FacetElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    if (!geomShapeFunc_) {
        // Fallback to linear Jacobian for unsupported geometry
        evalJacobianLinear();
        return;
    }
    
    // Use geometric shape functions for Jacobian computation
    ShapeValues sv = geomShapeFunc_->eval(&ip_.xi);
    
    // J = sum_k (x_k \otimes grad_xi(phi_k))
    jacobian_.setZero(spaceDim_, dim_);
    
    for (int j = 0; j < geomShapeFunc_->numDofs(); ++j) {
        for (int d1 = 0; d1 < spaceDim_; ++d1) {
            for (int d2 = 0; d2 < dim_; ++d2) {
                jacobian_(d1, d2) += vertices_[j][d1] * sv.gradients[j][d2];
            }
        }
    }
    
    evalState_ |= JACOBIAN_MASK;
}

void FacetElementTransform::evalJacobianLinear() const {
    // Linear Jacobian computation (fallback)
    switch (geometry_) {
        case Geometry::Segment: {
            for (int i = 0; i < spaceDim_; ++i) {
                jacobian_(i, 0) = 0.5 * (vertices_[1][i] - vertices_[0][i]);
            }
            break;
        }
        
        case Geometry::Triangle: {
            for (int i = 0; i < spaceDim_; ++i) {
                jacobian_(i, 0) = vertices_[1][i] - vertices_[0][i];
                jacobian_(i, 1) = vertices_[2][i] - vertices_[0][i];
            }
            break;
        }
        
        case Geometry::Square: {
            Real xi = ip_.xi, eta = ip_.eta;
            
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
    
    evalJacobian();
    
    // For 1D element: detJ = magnitude of tangent
    // For 2D element: compute area factor
    if (dim_ == 1) {
        // Segment in 2D or 3D space
        detJ_ = 0.0;
        for (int i = 0; i < spaceDim_; ++i) {
            detJ_ += jacobian_(i, 0) * jacobian_(i, 0);
        }
        detJ_ = std::sqrt(detJ_);
    } else if (dim_ == 2) {
        // Triangle or Square in 3D space
        // detJ = magnitude of normal vector = |J(:,0) x J(:,1)|
        Vector3 tangent1, tangent2;
        for (int i = 0; i < spaceDim_; ++i) {
            tangent1[i] = jacobian_(i, 0);
            tangent2[i] = jacobian_(i, 1);
        }
        Vector3 n = tangent1.cross(tangent2);
        detJ_ = n.norm();
    }
    
    weight_ = detJ_;
    evalState_ |= WEIGHT_MASK;
}

// =============================================================================
// Normal vector computation
// =============================================================================

Vector3 FacetElementTransform::normal() const {
    evalJacobian();
    
    Vector3 n;
    
    if (dim_ == 1) {
        // Segment in 2D or 3D space
        // Normal is perpendicular to tangent
        Vector3 tangent;
        for (int i = 0; i < spaceDim_; ++i) {
            tangent[i] = jacobian_(i, 0);
        }
        tangent.normalize();
        
        if (spaceDim_ == 2) {
            // Segment lies in a plane, normal is in that plane
            n = Vector3(-tangent.y(), tangent.x(), 0.0);
        } else if (spaceDim_ == 3) {
            // Segment is embedded in 3D, need reference direction
            // Use z-direction as reference
            Vector3 ref(0, 0, 1);
            n = tangent.cross(ref);
            if (n.norm() < 1e-10) {
                // Tangent is parallel to z-axis, use x-direction
                ref = Vector3(1, 0, 0);
                n = tangent.cross(ref);
            }
        }
    } else if (dim_ == 2) {
        // Triangle or Square in 3D space
        // Normal = J(:,0) x J(:,1)
        Vector3 tangent1, tangent2;
        for (int i = 0; i < spaceDim_; ++i) {
            tangent1[i] = jacobian_(i, 0);
            tangent2[i] = jacobian_(i, 1);
        }
        n = tangent1.cross(tangent2);
    }
    
    n.normalize();
    return n;
}

}  // namespace mpfem
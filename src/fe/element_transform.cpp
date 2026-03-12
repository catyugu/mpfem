#include "fe/element_transform.hpp"
#include "fe/shape_function.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// ElementTransform Implementation
// =============================================================================

Index ElementTransform::elementAttribute() const {
    if (!mesh_) return 0;
    
    if (elemIdx_ < mesh_->numElements()) {
        return mesh_->element(elemIdx_).attribute();
    }
    return 0;
}

void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (elemIdx_ < mesh_->numElements()) {
        elem = &mesh_->element(elemIdx_);
    }
    
    if (!elem) return;
    
    geometry_ = elem->geometry();
    dim_ = geom::dim(geometry_);
    
    // Get geometric order from the element
    geomOrder_ = elem->order();
    
    // Initialize geometric shape function
    initGeometricShapeFunction();
    
    // Get geometric node coordinates
    // For all nodes (corners + edge midpoints for quadratic elements)
    const std::vector<Index>& nodeIndices = elem->vertices();
    geomNodeIndices_.resize(nodeIndices.size());
    geomNodes_.resize(nodeIndices.size());
    
    for (size_t i = 0; i < nodeIndices.size(); ++i) {
        geomNodeIndices_[i] = nodeIndices[i];
        geomNodes_[i] = mesh_->vertex(nodeIndices[i]).toVector();
    }
    
    // Pre-allocate matrices
    jacobian_.setZero(dim_, dim_);
    invJacobian_.setZero(dim_, dim_);
    adjJacobian_.setZero(dim_, dim_);
}

void ElementTransform::initGeometricShapeFunction() {
    // Create geometric shape function based on geometry type and order
    // This is used for coordinate transformation and Jacobian computation
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
        case Geometry::Tetrahedron:
            geomShapeFunc_ = std::make_unique<H1TetrahedronShape>(geomOrder_);
            break;
        case Geometry::Cube:
            geomShapeFunc_ = std::make_unique<H1CubeShape>(geomOrder_);
            break;
        default:
            geomShapeFunc_.reset();
            break;
    }
}

// =============================================================================
// Jacobian evaluation - using geometric shape functions
// =============================================================================

void ElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    if (!geomShapeFunc_ || geomNodes_.empty()) {
        evalState_ |= JACOBIAN_MASK;
        return;
    }
    
    // Evaluate geometric shape functions and their gradients at current point
    geomShapeValues_ = geomShapeFunc_->eval(&ip_.xi);
    
    // Compute Jacobian: J = sum_i x_i (x) grad_xi(phi_i)
    // where x_i is the i-th geometric node coordinate
    // and grad_xi(phi_i) is the gradient of the i-th geometric shape function
    jacobian_.setZero(dim_, dim_);
    
    for (int i = 0; i < geomShapeFunc_->numDofs() && i < static_cast<int>(geomNodes_.size()); ++i) {
        const Vector3& nodeCoord = geomNodes_[i];
        const Vector3& shapeGrad = geomShapeValues_.gradients[i];
        
        // J += x_i (x) grad_phi_i
        // J(row, col) = x_i[row] * grad_phi_i[col]
        for (int row = 0; row < dim_; ++row) {
            for (int col = 0; col < dim_; ++col) {
                jacobian_(row, col) += nodeCoord[row] * shapeGrad[col];
            }
        }
    }
    
    evalState_ |= JACOBIAN_MASK;
}

void ElementTransform::evalWeight() const {
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

void ElementTransform::evalAdjugate() const {
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

void ElementTransform::evalInverse() const {
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

void ElementTransform::evalInvJacobianT() const {
    if (evalState_ & INV_JACOBIAN_T_MASK) return;
    
    // Ensure inverse Jacobian is computed
    evalInverse();
    
    // invJacobianT_ = invJacobian_^T
    invJacobianT_ = invJacobian_.transpose();
    
    evalState_ |= INV_JACOBIAN_T_MASK;
}

// =============================================================================
// Transform implementations - using geometric shape functions
// =============================================================================

void ElementTransform::transform(const Real* xi, Real* x) const {
    x[0] = x[1] = x[2] = 0.0;
    
    if (!geomShapeFunc_ || geomNodes_.empty()) return;
    
    // Evaluate geometric shape function values
    std::vector<Real> shapeVals = geomShapeFunc_->evalValues(xi);
    
    // x = sum_i phi_i(xi) * x_i
    for (int i = 0; i < geomShapeFunc_->numDofs() && i < static_cast<int>(geomNodes_.size()); ++i) {
        for (int d = 0; d < 3; ++d) {
            x[d] += shapeVals[i] * geomNodes_[i][d];
        }
    }
}

void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    // grad_x(phi) = J^{-T} * grad_xi(phi)
    // Using cached invJacobianT_ for efficiency
    
    const Matrix& invJT = invJacobianT();
    
    for (int i = 0; i < dim_; ++i) {
        physGrad[i] = 0.0;
        for (int j = 0; j < dim_; ++j) {
            physGrad[i] += invJT(i, j) * refGrad[j];
        }
    }
}

}  // namespace mpfem
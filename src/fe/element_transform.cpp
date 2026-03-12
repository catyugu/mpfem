#include "fe/element_transform.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// ElementTransform Implementation
// =============================================================================

void ElementTransform::setMesh(const Mesh* mesh) {
    mesh_ = mesh;
    evalState_ = 0;
}

void ElementTransform::setElement(Index elemIdx) {
    elemIdx_ = elemIdx;
    evalState_ = 0;
    computeGeometryInfo();
}

Index ElementTransform::attribute() const {
    if (!mesh_) return 0;
    
    if (elemType_ == VOLUME && elemIdx_ < mesh_->numElements()) {
        return mesh_->element(elemIdx_).attribute();
    } else if (elemType_ == BOUNDARY && elemIdx_ < mesh_->numBdrElements()) {
        return mesh_->bdrElement(elemIdx_).attribute();
    }
    return 0;
}

void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (elemType_ == VOLUME) {
        if (elemIdx_ >= mesh_->numElements()) return;
        elem = &mesh_->element(elemIdx_);
    } else {
        if (elemIdx_ >= mesh_->numBdrElements()) return;
        elem = &mesh_->bdrElement(elemIdx_);
    }
    
    geometry_ = elem->geometry();
    spaceDim_ = mesh_->dim();
    dim_ = geom::dim(geometry_);
    geomOrder_ = elem->order();
    
    // Get node coordinates
    nodeIndices_ = elem->vertices();
    nodes_.resize(nodeIndices_.size());
    for (size_t i = 0; i < nodeIndices_.size(); ++i) {
        nodes_[i] = mesh_->vertex(nodeIndices_[i]).toVector();
    }
    
    // Pre-allocate matrices
    jacobian_.setZero(spaceDim_, dim_);
    invJacobian_.setZero(dim_, spaceDim_);
    invJacobianT_.setZero(spaceDim_, dim_);
    adjJacobian_.setZero(spaceDim_, dim_);
    
    initGeometricShapeFunction();
}

void ElementTransform::initGeometricShapeFunction() {
    shapeFunc_ = ShapeFunction::create(geometry_, geomOrder_);
}

void ElementTransform::transform(const Real* xi, Real* x) const {
    if (!shapeFunc_ || shapeValues_.values.empty()) {
        // Fallback: use linear interpolation for vertices
        for (int d = 0; d < spaceDim_; ++d) x[d] = 0.0;
        return;
    }
    
    // Recompute shape values if needed
    ShapeValues sv;
    if (shapeValues_.values.empty()) {
        IntegrationPoint ip;
        ip.xi = xi[0];
        if (dim_ > 1) ip.eta = xi[1];
        if (dim_ > 2) ip.zeta = xi[2];
        sv = shapeFunc_->eval(ip);
    } else {
        sv = shapeValues_;
    }
    
    for (int d = 0; d < spaceDim_; ++d) {
        x[d] = 0.0;
        for (size_t i = 0; i < sv.values.size(); ++i) {
            x[d] += sv.values[i] * nodes_[i][d];
        }
    }
}

void ElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    if (!shapeFunc_ || shapeValues_.gradients.empty()) {
        jacobian_.setZero();
        evalState_ |= JACOBIAN_MASK;
        return;
    }
    
    // J = sum_i (x_i * grad_phi_i^T)
    jacobian_.setZero(spaceDim_, dim_);
    for (size_t i = 0; i < shapeValues_.gradients.size(); ++i) {
        const auto& grad = shapeValues_.gradients[i];
        for (int d = 0; d < spaceDim_; ++d) {
            for (int k = 0; k < dim_; ++k) {
                jacobian_(d, k) += nodes_[i][d] * grad[k];
            }
        }
    }
    
    evalState_ |= JACOBIAN_MASK;
}

void ElementTransform::evalWeight() const {
    if (evalState_ & WEIGHT_MASK) return;
    
    evalJacobian();
    
    if (dim_ == spaceDim_) {
        // Square Jacobian: weight = |det(J)|
        detJ_ = jacobian_.determinant();
        weight_ = std::abs(detJ_);
    } else {
        // Non-square Jacobian (boundary element): weight = sqrt(det(J^T J))
        Matrix JtJ = jacobian_.transpose() * jacobian_;
        detJ_ = JtJ.determinant();
        weight_ = std::sqrt(std::abs(detJ_));
    }
    
    evalState_ |= WEIGHT_MASK;
}

void ElementTransform::evalAdjugate() const {
    if (evalState_ & ADJUGATE_MASK) return;
    
    evalJacobian();
    evalWeight();
    
    if (dim_ == spaceDim_ && dim_ > 0) {
        // adj(J) = det(J) * J^{-1} = cofactor matrix
        adjJacobian_ = jacobian_.inverse() * detJ_;
    } else {
        // For non-square, use pseudo-inverse approach
        adjJacobian_ = jacobian_;
    }
    
    evalState_ |= ADJUGATE_MASK;
}

void ElementTransform::evalInverse() const {
    if (evalState_ & INVERSE_MASK) return;
    
    evalJacobian();
    evalWeight();
    
    if (dim_ == spaceDim_ && std::abs(detJ_) > 1e-15) {
        invJacobian_ = jacobian_.inverse();
    } else if (dim_ < spaceDim_) {
        // Use pseudo-inverse: J^+ = (J^T J)^{-1} J^T
        Matrix JtJ = jacobian_.transpose() * jacobian_;
        invJacobian_ = JtJ.ldlt().solve(jacobian_.transpose());
    }
    
    evalState_ |= INVERSE_MASK;
}

void ElementTransform::evalInvJacobianT() const {
    if (evalState_ & INV_JACOBIAN_T_MASK) return;
    
    evalInverse();
    invJacobianT_ = invJacobian_.transpose();
    
    evalState_ |= INV_JACOBIAN_T_MASK;
}

Real ElementTransform::weight() const {
    evalWeight();
    return weight_;
}

void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    evalInvJacobianT();
    
    // physGrad = J^{-T} * refGrad
    for (int d = 0; d < spaceDim_; ++d) {
        physGrad[d] = 0.0;
        for (int k = 0; k < dim_; ++k) {
            physGrad[d] += invJacobianT_(d, k) * refGrad[k];
        }
    }
}

}  // namespace mpfem

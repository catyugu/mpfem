#include "fe/element_transform.hpp"
#include "core/exception.hpp"
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
    if (!mesh_) {
        MPFEM_THROW(Exception, "ElementTransform::attribute: mesh not set");
    }
    
    if (elemType_ == VOLUME) {
        if (elemIdx_ >= mesh_->numElements()) {
            MPFEM_THROW(RangeException, 
                "ElementTransform::attribute: invalid element index " + 
                std::to_string(elemIdx_) + ", num elements = " + 
                std::to_string(mesh_->numElements()));
        }
        return mesh_->element(elemIdx_).attribute();
    } else if (elemType_ == BOUNDARY) {
        if (elemIdx_ >= mesh_->numBdrElements()) {
            MPFEM_THROW(RangeException, 
                "ElementTransform::attribute: invalid boundary element index " + 
                std::to_string(elemIdx_) + ", num boundary elements = " + 
                std::to_string(mesh_->numBdrElements()));
        }
        return mesh_->bdrElement(elemIdx_).attribute();
    }
    
    MPFEM_THROW(Exception, "ElementTransform::attribute: unknown element type");
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
    
    // Pre-allocate storage for shape function evaluation (avoids runtime allocation during assembly)
    if (shapeFunc_) {
        const int numDofs = shapeFunc_->numDofs();
        shapeValuesOnly_.resize(numDofs);
        shapeGradsOnly_.resize(numDofs);
    }
}

void ElementTransform::initGeometricShapeFunction() {
    shapeFunc_ = ShapeFunction::create(geometry_, geomOrder_);
}

void ElementTransform::transform(const Real* xi, Real* x) const {
    if (!shapeFunc_) {
        MPFEM_THROW(Exception, "ElementTransform::transform: shape function not initialized");
    }
    
    // Evaluate only shape function values (no gradients needed for coordinate transform)
    shapeFunc_->evalValues(xi, shapeValuesOnly_.data());
    
    for (int d = 0; d < spaceDim_; ++d) {
        x[d] = 0.0;
        for (size_t i = 0; i < shapeValuesOnly_.size(); ++i) {
            x[d] += shapeValuesOnly_[i] * nodes_[i][d];
        }
    }
}

void ElementTransform::evalJacobian() const {
    if (evalState_ & JACOBIAN_MASK) return;
    
    if (!shapeFunc_) {
        MPFEM_THROW(Exception, "ElementTransform::evalJacobian: shape function not initialized");
    }
    
    // Evaluate only shape function gradients (no values needed for Jacobian)
    shapeFunc_->evalGrads(&ip_.xi, shapeGradsOnly_.data());
    
    // J = sum_i (x_i * grad_phi_i^T)
    jacobian_.setZero(spaceDim_, dim_);
    for (size_t i = 0; i < shapeGradsOnly_.size(); ++i) {
        const auto& grad = shapeGradsOnly_[i];
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
        // Non-square Jacobian (boundary element):
        // weight = sqrt(det(J^T J)) = area/volume scaling factor
        // detJ should also be this value for consistency
        Matrix JtJ = jacobian_.transpose() * jacobian_;
        weight_ = std::sqrt(std::abs(JtJ.determinant()));
        detJ_ = weight_;
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

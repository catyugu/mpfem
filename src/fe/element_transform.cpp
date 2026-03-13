#include "fe/element_transform.hpp"
#include "core/exception.hpp"
#include <cmath>
#include <iostream>

namespace mpfem {

void ElementTransform::setMesh(const Mesh* mesh) {
    mesh_ = mesh;
    computeGeometryInfo();
}

void ElementTransform::setElement(Index elemIdx) {
    elemIdx_ = elemIdx;
    computeGeometryInfo();
}

Index ElementTransform::attribute() const {
    if (!mesh_) {
        MPFEM_THROW(Exception, "ElementTransform::attribute: mesh not set");
    }
    
    if (elemType_ == VOLUME) {
        if (elemIdx_ >= mesh_->numElements()) {
            MPFEM_THROW(RangeException, 
                "ElementTransform::attribute: invalid element index");
        }
        return mesh_->element(elemIdx_).attribute();
    } else {
        if (elemIdx_ >= mesh_->numBdrElements()) {
            MPFEM_THROW(RangeException, 
                "ElementTransform::attribute: invalid boundary element index");
        }
        return mesh_->bdrElement(elemIdx_).attribute();
    }
}

void ElementTransform::computeGeometryInfo() {
    if (!mesh_) return;
    
    const Element* elem = nullptr;
    if (elemType_ == VOLUME) {
        if (elemIdx_ < 0 || elemIdx_ >= static_cast<Index>(mesh_->numElements())) return;
        elem = &mesh_->element(elemIdx_);
    } else {
        if (elemIdx_ < 0 || elemIdx_ >= static_cast<Index>(mesh_->numBdrElements())) return;
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
    
    // Create shape function
    shapeFunc_ = ShapeFunction::create(geometry_, geomOrder_);
    
    // Pre-allocate shape function buffers
    if (shapeFunc_) {
        const int numDofs = shapeFunc_->numDofs();
        shapeValuesBuf_.resize(numDofs);
        shapeGradsBuf_.resize(numDofs);
    }
}

void ElementTransform::computeJacobianAtIP() {
    if (!shapeFunc_) return;
    
    // Evaluate shape function gradients at integration point
    shapeFunc_->evalGrads(&ip_.xi, shapeGradsBuf_.data());
    
    // Compute Jacobian: J = sum_i (x_i * grad_phi_i^T)
    jacobian_.setZero(spaceDim_, dim_);
    for (size_t i = 0; i < shapeGradsBuf_.size(); ++i) {
        const auto& grad = shapeGradsBuf_[i];
        for (int d = 0; d < spaceDim_; ++d) {
            for (int k = 0; k < dim_; ++k) {
                jacobian_(d, k) += nodes_[i][d] * grad[k];
            }
        }
    }
    
    // Compute weight
    if (dim_ == spaceDim_) {
        detJ_ = jacobian_.determinant();
        weight_ = std::abs(detJ_);
    } else {
        Matrix JtJ = jacobian_.transpose() * jacobian_;
        weight_ = std::sqrt(std::abs(JtJ.determinant()));
        detJ_ = weight_;
    }
    
    // Compute inverse Jacobian
    if (dim_ == spaceDim_ && std::abs(detJ_) > 1e-15) {
        invJacobian_ = jacobian_.inverse();
        adjJacobian_ = invJacobian_ * detJ_;
    } else if (dim_ < spaceDim_) {
        Matrix JtJ = jacobian_.transpose() * jacobian_;
        invJacobian_ = JtJ.ldlt().solve(jacobian_.transpose());
        adjJacobian_ = jacobian_;
    }
    
    // Compute transpose
    invJacobianT_ = invJacobian_.transpose();
}

void ElementTransform::transform(const Real* xi, Real* x) const {
    if (!shapeFunc_) return;
    
    // Evaluate shape values
    std::vector<Real> vals(shapeValuesBuf_.size());
    shapeFunc_->evalValues(xi, vals.data());
    
    for (int d = 0; d < spaceDim_; ++d) {
        x[d] = 0.0;
        for (size_t i = 0; i < vals.size(); ++i) {
            x[d] += vals[i] * nodes_[i][d];
        }
    }
}

void ElementTransform::transformGradient(const Real* refGrad, Real* physGrad) const {
    // physGrad = J^{-T} * refGrad
    for (int d = 0; d < spaceDim_; ++d) {
        physGrad[d] = 0.0;
        for (int k = 0; k < dim_; ++k) {
            physGrad[d] += invJacobianT_(d, k) * refGrad[k];
        }
    }
}

}  // namespace mpfem
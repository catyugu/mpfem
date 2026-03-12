#include "fe/grid_function.hpp"
#include "fe/element_transform.hpp"
#include "fe/shape_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

GridFunction::GridFunction(const FESpace* fes) 
    : fes_(fes) {
    if (fes_) {
        values_.resize(fes_->numDofs());
        values_.setZero();
    }
}

GridFunction::GridFunction(const FESpace* fes, Real initValue)
    : fes_(fes) {
    if (fes_) {
        values_.resize(fes_->numDofs());
        values_.setConstant(initValue);
    }
}

void GridFunction::setValues(const Real* data, Index size) {
    values_.resize(size);
    for (Index i = 0; i < size; ++i) {
        values_[i] = data[i];
    }
}

void GridFunction::getElementValues(Index elemIdx, std::vector<Real>& values) const {
    if (!fes_) {
        values.clear();
        return;
    }
    
    fes_->getElementDofs(elemIdx, dofCache_);
    values.resize(dofCache_.size());
    
    for (size_t i = 0; i < dofCache_.size(); ++i) {
        values[i] = values_[dofCache_[i]];
    }
}

Eigen::VectorXd GridFunction::getElementValues(Index elemIdx) const {
    if (!fes_) return Eigen::VectorXd();
    
    fes_->getElementDofs(elemIdx, dofCache_);
    Eigen::VectorXd localValues(dofCache_.size());
    
    for (size_t i = 0; i < dofCache_.size(); ++i) {
        localValues[i] = values_[dofCache_[i]];
    }
    
    return localValues;
}

void GridFunction::addElementValues(Index elemIdx, const Eigen::VectorXd& localValues, Real scale) {
    if (!fes_) return;
    
    fes_->getElementDofs(elemIdx, dofCache_);
    
    for (size_t i = 0; i < dofCache_.size() && i < static_cast<size_t>(localValues.size()); ++i) {
        values_[dofCache_[i]] += scale * localValues[i];
    }
}

void GridFunction::setElementValues(Index elemIdx, const Eigen::VectorXd& localValues) {
    if (!fes_) return;
    
    fes_->getElementDofs(elemIdx, dofCache_);
    
    for (size_t i = 0; i < dofCache_.size() && i < static_cast<size_t>(localValues.size()); ++i) {
        values_[dofCache_[i]] = localValues[i];
    }
}

Real GridFunction::eval(Index elemIdx, const Real* xi) const {
    if (!fes_) return 0.0;
    
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) return 0.0;
    
    // Get shape function values
    auto shapeValues = refElem->shapeFunction()->evalValues(xi);
    
    // Get element DOFs
    fes_->getElementDofs(elemIdx, dofCache_);
    
    // Interpolate: u(xi) = sum_i phi_i(xi) * u_i
    Real value = 0.0;
    int nd = std::min(static_cast<int>(shapeValues.size()), 
                      static_cast<int>(dofCache_.size()));
    
    for (int i = 0; i < nd; ++i) {
        value += shapeValues[i] * values_[dofCache_[i]];
    }
    
    return value;
}

Vector3 GridFunction::evalVector(Index elemIdx, const Real* xi) const {
    if (!fes_ || fes_->vdim() == 1) {
        return Vector3(eval(elemIdx, xi), 0.0, 0.0);
    }
    
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) return Vector3::Zero();
    
    auto shapeValues = refElem->shapeFunction()->evalValues(xi);
    fes_->getElementDofs(elemIdx, dofCache_);
    
    int nd = refElem->numDofs();
    int vdim = fes_->vdim();
    
    Vector3 result = Vector3::Zero();
    
    for (int c = 0; c < vdim; ++c) {
        Real compValue = 0.0;
        for (int i = 0; i < nd; ++i) {
            Index dofIdx = c * nd + i;
            if (dofIdx < static_cast<Index>(dofCache_.size())) {
                compValue += shapeValues[i] * values_[dofCache_[dofIdx]];
            }
        }
        result[c] = compValue;
    }
    
    return result;
}

Vector3 GridFunction::gradient(Index elemIdx, const Real* xi, ElementTransform& trans) const {
    if (!fes_) return Vector3::Zero();
    
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) return Vector3::Zero();
    
    // Get shape function values and gradients in reference coordinates
    ShapeValues shapeVals = refElem->shapeFunction()->eval(xi);
    
    // Get element DOFs
    fes_->getElementDofs(elemIdx, dofCache_);
    
    // Compute gradient in reference coordinates
    Vector3 gradRef = Vector3::Zero();
    int nd = std::min(static_cast<int>(shapeVals.size()),
                      static_cast<int>(dofCache_.size()));
    
    for (int i = 0; i < nd; ++i) {
        gradRef += shapeVals.gradients[i] * values_[dofCache_[i]];
    }
    
    // Transform to physical coordinates using inverse Jacobian
    // grad_physical = J^{-T} * grad_reference
    // Set integration point first, then get Jacobian
    trans.setIntegrationPoint(xi);

    const Matrix& JinvT = trans.invJacobianT();
    
    return JinvT * gradRef;
}

}  // namespace mpfem

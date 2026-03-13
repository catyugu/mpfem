#include "fe/grid_function.hpp"
#include "fe/element_transform.hpp"
#include "fe/shape_function.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"

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
    
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    values.resize(dofs.size());
    
    for (size_t i = 0; i < dofs.size(); ++i) {
        values[i] = values_[dofs[i]];
    }
}

Eigen::VectorXd GridFunction::getElementValues(Index elemIdx) const {
    if (!fes_) return Eigen::VectorXd();
    
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    Eigen::VectorXd localValues(dofs.size());
    
    for (size_t i = 0; i < dofs.size(); ++i) {
        localValues[i] = values_[dofs[i]];
    }
    
    return localValues;
}

void GridFunction::addElementValues(Index elemIdx, const Eigen::VectorXd& localValues, Real scale) {
    if (!fes_) return;
    
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    
    for (size_t i = 0; i < dofs.size() && i < static_cast<size_t>(localValues.size()); ++i) {
        values_[dofs[i]] += scale * localValues[i];
    }
}

void GridFunction::setElementValues(Index elemIdx, const Eigen::VectorXd& localValues) {
    if (!fes_) return;
    
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    
    for (size_t i = 0; i < dofs.size() && i < static_cast<size_t>(localValues.size()); ++i) {
        values_[dofs[i]] = localValues[i];
    }
}

Real GridFunction::eval(Index elemIdx, const Real* xi) const {
    if (!fes_) {
        MPFEM_THROW(Exception, "GridFunction::eval: FE space not set");
    }
    
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) {
        MPFEM_THROW(Exception, 
            "GridFunction::eval: no reference element for element " + 
            std::to_string(elemIdx));
    }
    
    // Get shape function values (pre-allocated storage)
    const ShapeFunction* shapeFunc = refElem->shapeFunction();
    const int numDofs = shapeFunc->numDofs();
    std::vector<Real> shapeValues(numDofs);
    shapeFunc->evalValues(xi, shapeValues.data());
    
    // Get element DOFs
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    
    // Interpolate: u(xi) = sum_i phi_i(xi) * u_i
    Real value = 0.0;
    int nd = std::min(numDofs, static_cast<int>(dofs.size()));
    
    for (int i = 0; i < nd; ++i) {
        value += shapeValues[i] * values_[dofs[i]];
    }
    
    return value;
}

Vector3 GridFunction::evalVector(Index elemIdx, const Real* xi) const {
    if (!fes_ || fes_->vdim() == 1) {
        return Vector3(eval(elemIdx, xi), 0.0, 0.0);
    }
    
    const ReferenceElement* refElem = fes_->elementRefElement(elemIdx);
    if (!refElem) return Vector3::Zero();
    
    // Get shape function values (pre-allocated storage)
    const ShapeFunction* shapeFunc = refElem->shapeFunction();
    const int numDofs = shapeFunc->numDofs();
    std::vector<Real> shapeValues(numDofs);
    shapeFunc->evalValues(xi, shapeValues.data());
    
    std::vector<Index> dofs;  // Thread-safe: local variable
    fes_->getElementDofs(elemIdx, dofs);
    
    int nd = refElem->numDofs();
    int vdim = fes_->vdim();
    
    Vector3 result = Vector3::Zero();
    
    for (int c = 0; c < vdim; ++c) {
        Real compValue = 0.0;
        for (int i = 0; i < nd; ++i) {
            Index dofIdx = c * nd + i;
            if (dofIdx < static_cast<Index>(dofs.size())) {
                compValue += shapeValues[i] * values_[dofs[dofIdx]];
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
    
    // Get shape function gradients in reference coordinates (only gradients needed)
    const ShapeFunction* shapeFunc = refElem->shapeFunction();
    const int numDofs = shapeFunc->numDofs();
    std::vector<Vector3> shapeGrads(numDofs);
    shapeFunc->evalGrads(xi, shapeGrads.data());
    
    // Get element DOFs (use local variable for thread safety)
    std::vector<Index> dofs;
    fes_->getElementDofs(elemIdx, dofs);
    
    // Compute gradient in reference coordinates
    Vector3 gradRef = Vector3::Zero();
    int nd = std::min(numDofs, static_cast<int>(dofs.size()));
    
    for (int i = 0; i < nd; ++i) {
        gradRef += shapeGrads[i] * values_[dofs[i]];
    }
    
    // Transform to physical coordinates using inverse Jacobian
    // grad_physical = J^{-T} * grad_reference
    // Set integration point first, then get Jacobian
    trans.setIntegrationPoint(xi);

    const Matrix& JinvT = trans.invJacobianT();
    
    return JinvT * gradRef;
}

}  // namespace mpfem

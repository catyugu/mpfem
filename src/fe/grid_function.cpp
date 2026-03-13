#include "grid_function.hpp"
#include "element_transform.hpp"
#include "shape_function.hpp"

namespace mpfem {

Real GridFunction::eval(Index elem, const Real* xi) const {
    if (!fes_) return 0.0;
    
    const ReferenceElement* ref = fes_->elementRefElement(elem);
    if (!ref) return 0.0;
    
    const ShapeFunction* sf = ref->shapeFunction();
    int nd = sf->numDofs();
    
    std::vector<Real> phi(nd);
    sf->evalValues(xi, phi.data());
    
    std::vector<Index> dofs;
    fes_->getElementDofs(elem, dofs);
    
    Real val = 0.0;
    for (int i = 0; i < nd && i < static_cast<int>(dofs.size()); ++i) {
        val += phi[i] * values_[dofs[i]];
    }
    return val;
}

Vector3 GridFunction::gradient(Index elem, const Real* xi, ElementTransform& trans) const {
    if (!fes_) return Vector3::Zero();
    
    const ReferenceElement* ref = fes_->elementRefElement(elem);
    if (!ref) return Vector3::Zero();
    
    const ShapeFunction* sf = ref->shapeFunction();
    int nd = sf->numDofs();
    
    std::vector<Vector3> grads(nd);
    sf->evalGrads(xi, grads.data());
    
    std::vector<Index> dofs;
    fes_->getElementDofs(elem, dofs);
    
    Vector3 gRef = Vector3::Zero();
    for (int i = 0; i < nd && i < static_cast<int>(dofs.size()); ++i) {
        gRef += grads[i] * values_[dofs[i]];
    }
    
    trans.setIntegrationPoint(xi);
    return trans.invJacobianT() * gRef;
}

}  // namespace mpfem

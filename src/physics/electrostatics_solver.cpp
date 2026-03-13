#include "electrostatics_solver.hpp"
#include "fe/coefficient.hpp"

namespace mpfem {

void ElectrostaticsSolver::setTemperatureField(const GridFunction *temperature) {
    temperatureField_ = temperature;
    // If conductivity is a TemperatureDependentConductivityCoefficient, set its temperature field
    if (auto* tempDepCond = dynamic_cast<TemperatureDependentConductivityCoefficient*>(conductivity_.get())) {
        tempDepCond->setTemperatureField(temperature);
    }
}

bool ElectrostaticsSolver::computeJouleHeat(std::vector<Real>& Q) const {
    if (!V_ || !mesh_ || !conductivity_) {
        return false;
    }
    
    Index numElements = mesh_->numElements();
    Q.resize(numElements);
    std::fill(Q.begin(), Q.end(), 0.0);
    
    ElementTransform trans;
    trans.setMesh(mesh_);
    
    for (Index e = 0; e < numElements; ++e) {
        trans.setElement(e);
        
        // Use element center for simplified evaluation
        Real xi[3] = {0.0, 0.0, 0.0};
        trans.setIntegrationPoint(xi);
        
        // Get element DOFs
        std::vector<Index> dofs;
        fes_->getElementDofs(e, dofs);
        
        // Compute gradient of V at element center
        Vector3 gradV = V_->gradient(e, xi, trans);
        
        // Get conductivity for this element
        Real sigma = conductivity_->eval(trans);
        
        // Q = sigma * |gradV|^2
        Real gradMagSq = gradV.x()*gradV.x() + gradV.y()*gradV.y() + gradV.z()*gradV.z();
        Q[e] = sigma * gradMagSq;
    }
    
    return true;
}

} // namespace mpfem

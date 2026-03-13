#include "coefficient.hpp"
#include "element_transform.hpp"
#include "grid_function.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// PWConstCoefficient
// =============================================================================

Real PWConstCoefficient::eval(ElementTransform& trans) const {
    int attr = static_cast<int>(trans.attribute());
    if (attr < 1 || attr > static_cast<int>(values_.size())) return 0.0;
    return values_[attr - 1];
}

// =============================================================================
// FunctionCoefficient
// =============================================================================

Real FunctionCoefficient::eval(ElementTransform& trans) const {
    Real xi[3];
    trans.transform(&trans.integrationPoint().xi, xi);
    return func_(xi[0], xi[1], xi[2]);
}

// =============================================================================
// GridFunctionCoefficient
// =============================================================================

Real GridFunctionCoefficient::eval(ElementTransform& trans) const {
    if (!gf_) return 0.0;
    const auto& ip = trans.integrationPoint();
    Real xi[3] = {ip.xi, ip.eta, ip.zeta};
    return gf_->eval(trans.elementIndex(), xi);
}

// =============================================================================
// TemperatureDependentConductivity
// =============================================================================

Real TemperatureDependentConductivity::eval(ElementTransform& trans) const {
    int domainId = static_cast<int>(trans.attribute());
    
    auto it = rho0_.find(domainId);
    if (it == rho0_.end()) return 1.0;
    
    Real rho0 = it->second;
    Real alpha = alpha_.at(domainId);
    Real tref = tref_.at(domainId);
    
    Real temp = tref;
    if (tempFunc_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        temp = tempFunc_(trans.elementIndex(), xi);
    }
    
    Real rho = rho0 * (1.0 + alpha * (temp - tref));
    return 1.0 / rho;
}

}  // namespace mpfem

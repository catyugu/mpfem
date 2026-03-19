#include "coefficient.hpp"
#include "element_transform.hpp"
#include "grid_function.hpp"
#include <cmath>

namespace mpfem {

Real FunctionCoefficient::eval(ElementTransform& trans, Real t) const {
    Real xi[3];
    trans.transform(&trans.integrationPoint().xi, xi);
    return func_(xi[0], xi[1], xi[2], t);
}

Real DomainMappedCoefficient::eval(ElementTransform& trans, Real t) const {
    const Coefficient* coef = get(static_cast<int>(trans.attribute()));
    return coef ? coef->eval(trans, t) : 0.0;
}

Real TemperatureDependentConductivity::eval(ElementTransform& trans, Real) const {
    Real temp = tref_;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        temp = T_->eval(trans.elementIndex(), &ip.xi);
    }
    Real factor = 1.0 + alpha_ * (temp - tref_);
    return 1.0 / (rho0_ * (factor > 0 ? factor : 1e-10));
}

Real JouleHeatCoefficient::eval(ElementTransform& trans, Real t) const {
    if (!V_ || !sigma_) return 0.0;
    Real sigma_val = sigma_->eval(trans, t);
    Vector3 g = V_->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
    return sigma_val * g.squaredNorm();
}

Real ThermalExpansionCoefficient::eval(ElementTransform& trans, Real) const {
    Real T = T_ref_;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        T = T_->eval(trans.elementIndex(), &ip.xi);
    }
    return alpha_T_ * (T - T_ref_);
}

}  // namespace mpfem
#include "coefficient.hpp"
#include "element_transform.hpp"
#include "grid_function.hpp"
#include <cmath>

namespace mpfem {
    
// =============================================================================
// FunctionCoefficient
// =============================================================================

Real FunctionCoefficient::eval(ElementTransform& trans, Real t) const {
    Real xi[3];
    trans.transform(&trans.integrationPoint().xi, xi);
    return func_(xi[0], xi[1], xi[2], t);
}

// =============================================================================
// DomainMappedCoefficient
// =============================================================================

Real DomainMappedCoefficient::eval(ElementTransform& trans, Real t) const {
    int attr = static_cast<int>(trans.attribute());
    const Coefficient* coef = get(attr);
    return coef ? coef->eval(trans, t) : 0.0;
}

// =============================================================================
// TemperatureDependentConductivity
// =============================================================================

Real TemperatureDependentConductivity::eval(ElementTransform& trans, Real) const {
    // 获取温度
    Real temp = tref_;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        temp = T_->eval(trans.elementIndex(), xi);
    }
    
    // 线性电阻率模型: rho = rho0 * (1 + alpha * (T - Tref))
    Real factor = 1.0 + alpha_ * (temp - tref_);
    
    // 数值保护：防止负电阻率或零电阻率
    if (factor <= 0.0) {
        factor = 1e-10;
    }
    
    Real rho = rho0_ * factor;
    return 1.0 / rho;
}

// =============================================================================
// JouleHeatCoefficient
// =============================================================================

Real JouleHeatCoefficient::eval(ElementTransform& trans, Real t) const {
    if (!V_ || !sigma_) return 0.0;
    
    // 先评估电导率，再计算梯度
    Real sigma_val = sigma_->eval(trans, t);
    Vector3 g = V_->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
    return sigma_val * g.squaredNorm();
}

// =============================================================================
// ThermalExpansionCoefficient
// =============================================================================

Real ThermalExpansionCoefficient::eval(ElementTransform& trans, Real) const {
    // 获取温度
    Real T = T_ref_;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        T = T_->eval(trans.elementIndex(), xi);
    }
    
    // 计算热膨胀应变
    Real delta_T = T - T_ref_;
    return alpha_T_ * delta_T;
}

}  // namespace mpfem
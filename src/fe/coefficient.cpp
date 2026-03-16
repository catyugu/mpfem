#include "coefficient.hpp"
#include "element_transform.hpp"
#include "grid_function.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// PWConstCoefficient
// =============================================================================

Real PWConstCoefficient::eval(ElementTransform& trans, Real) const {
    int attr = static_cast<int>(trans.attribute());
    if (attr < 1 || attr > static_cast<int>(values_.size())) return 0.0;
    return values_[attr - 1];
}

// =============================================================================
// FunctionCoefficient
// =============================================================================

Real FunctionCoefficient::eval(ElementTransform& trans, Real t) const {
    Real xi[3];
    trans.transform(&trans.integrationPoint().xi, xi);
    return func_(xi[0], xi[1], xi[2], t);
}

// =============================================================================
// GridFunctionCoefficient
// =============================================================================

Real GridFunctionCoefficient::eval(ElementTransform& trans, Real) const {
    if (!gf_) return 0.0;
    const auto& ip = trans.integrationPoint();
    Real xi[3] = {ip.xi, ip.eta, ip.zeta};
    return gf_->eval(trans.elementIndex(), xi);
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
    int attr = static_cast<int>(trans.attribute());
    if (attr < 1 || attr > static_cast<int>(rho0_.size())) return 1.0;
    
    Real rho0 = rho0_[attr - 1];
    
    // rho0 <= 0 表示使用常量电导率
    if (rho0 <= 0.0) {
        return sigma0_[attr - 1];
    }
    
    Real alpha = alpha_[attr - 1];
    Real tref = tref_[attr - 1];
    
    // 获取温度
    Real temp = tref;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        temp = T_->eval(trans.elementIndex(), xi);
    }
    
    // 线性电阻率模型: rho = rho0 * (1 + alpha * (T - Tref))
    Real factor = 1.0 + alpha * (temp - tref);
    
    // 数值保护：防止负电阻率或零电阻率
    if (factor <= 0.0) {
        factor = 1e-10;
    }
    
    Real rho = rho0 * factor;
    return 1.0 / rho;
}

// =============================================================================
// JouleHeatCoefficient
// =============================================================================

Real JouleHeatCoefficient::eval(ElementTransform& trans, Real t) const {
    if (!V_ || !sigma_) return 0.0;
    
    // 检查域限制
    if (!domains_.empty()) {
        int attr = static_cast<int>(trans.attribute());
        if (domains_.find(attr) == domains_.end()) return 0.0;
    }
    
    // 先评估电导率，再计算梯度
    Real sigma_val = sigma_->eval(trans, t);
    Vector3 g = V_->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
    return sigma_val * g.squaredNorm();
}

// =============================================================================
// ThermalExpansionCoefficient
// =============================================================================

Real ThermalExpansionCoefficient::eval(ElementTransform& trans, Real) const {
    int attr = static_cast<int>(trans.attribute());
    if (attr < 1 || attr > static_cast<int>(alpha_T_.size())) return 0.0;
    
    Real alpha_T = alpha_T_[attr - 1];
    if (alpha_T == 0.0) return 0.0;
    
    // 获取温度
    Real T = T_ref_;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        T = T_->eval(trans.elementIndex(), xi);
    }
    
    // 计算热膨胀应变
    Real delta_T = T - T_ref_;
    return alpha_T * delta_T;
}

}  // namespace mpfem
#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <functional>
#include <vector>
#include <set>
#include <map>

namespace mpfem {

class ElementTransform;
class GridFunction;

// =============================================================================
// 标量系数基类
// =============================================================================

/**
 * @brief 标量系数基类
 * 
 * 设计原则：
 * - 使用 ElementTransform 获取几何和域信息
 * - 支持时间参数 t 用于瞬态问题
 * - 纯虚接口，派生类必须实现 eval
 * 
 * 所有权策略：
 * - Coefficient 实例由调用者（如 PhysicsProblemSetup）管理
 * - DomainMappedCoefficient 只持有非拥有引用
 */
class Coefficient {
public:
    virtual ~Coefficient() = default;
    
    /**
     * @brief 在积分点评估系数值
     * @param trans 单元变换（包含积分点、域ID等信息）
     * @param t 时间参数（默认为0，用于瞬态问题）
     * @return 系数值
     */
    virtual Real eval(ElementTransform& trans, Real t = 0.0) const = 0;
};

// =============================================================================
// 基础系数
// =============================================================================

/// 常量系数
class ConstantCoefficient : public Coefficient {
public:
    explicit ConstantCoefficient(Real c = 1.0) : value_(c) {}
    Real eval(ElementTransform&, Real = 0.0) const override { return value_; }
    void set(Real c) { value_ = c; }
    Real get() const { return value_; }
private:
    Real value_;
};

/// 函数系数
class FunctionCoefficient : public Coefficient {
public:
    using Func = std::function<Real(Real, Real, Real, Real)>;  // x, y, z, t
    explicit FunctionCoefficient(Func f) : func_(std::move(f)) {}
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    Func func_;
};

/// 域映射系数：不同域使用不同的系数（非持有引用）
/// 
/// 所有权策略：所有系数由外部管理（如 PhysicsProblemSetup::coefficients_）
class DomainMappedCoefficient : public Coefficient {
public:
    DomainMappedCoefficient() = default;
    
    /// 移动构造函数
    DomainMappedCoefficient(DomainMappedCoefficient&& other) noexcept
        : coefs_(std::move(other.coefs_)), defaultCoef_(other.defaultCoef_) {
        other.defaultCoef_ = nullptr;
    }
    
    /// 移动赋值运算符
    DomainMappedCoefficient& operator=(DomainMappedCoefficient&& other) noexcept {
        if (this != &other) {
            coefs_ = std::move(other.coefs_);
            defaultCoef_ = other.defaultCoef_;
            other.defaultCoef_ = nullptr;
        }
        return *this;
    }
    
    /// 设置指定域的系数（非持有指针）
    void set(int domainId, const Coefficient* coef) {
        coefs_[domainId] = coef;
    }
    
    /// 批量设置多个域使用同一个系数（非持有指针）
    void set(const std::set<int>& domainIds, const Coefficient* coef) {
        for (int id : domainIds) {
            coefs_[id] = coef;
        }
    }
    
    /// 设置所有域使用同一个系数（非持有指针，清空映射，设置默认系数）
    void setAll(const Coefficient* coef) {
        defaultCoef_ = coef;
        coefs_.clear();
    }
    
    /// 获取指定域的系数（如果没有则返回默认系数）
    const Coefficient* get(int domainId) const {
        auto it = coefs_.find(domainId);
        if (it != coefs_.end()) return it->second;
        return defaultCoef_;
    }
    
    /// 评估系数值（实现在 cpp 文件中）
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
    /// 检查是否有任何系数设置
    bool empty() const { return coefs_.empty() && defaultCoef_ == nullptr; }
    
private:
    std::map<int, const Coefficient*> coefs_;
    const Coefficient* defaultCoef_ = nullptr;
};

// =============================================================================
// 温度依赖电导率系数
// =============================================================================

/**
 * @brief 温度依赖电导率：sigma = 1 / (rho0 * (1 + alpha * (T - Tref)))
 * 
 * 单材料参数设计：每个实例仅存储一种材料的参数。
 * 多域场景通过 DomainMappedCoefficient 组合。
 */
class TemperatureDependentConductivity : public Coefficient {
public:
    /// 构造时注入温度场引用和材料参数
    TemperatureDependentConductivity(const GridFunction& T,
                                      Real rho0, Real alpha, Real tref)
        : T_(&T), rho0_(rho0), alpha_(alpha), tref_(tref) {}
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    const GridFunction* T_;
    Real rho0_;
    Real alpha_;
    Real tref_;
};

// =============================================================================
// 焦耳热系数
// =============================================================================

/**
 * @brief 焦耳热系数: Q = sigma * |grad V|^2
 * 
 * 在构造时注入电势场和电导率引用。
 * 域限制通过 DomainMappedCoefficient 实现。
 */
class JouleHeatCoefficient : public Coefficient {
public:
    /// 构造时注入电势场和电导率引用
    JouleHeatCoefficient(const GridFunction& V, const Coefficient& sigma)
        : V_(&V), sigma_(&sigma) {}
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    const GridFunction* V_;
    const Coefficient* sigma_;
};

// =============================================================================
// 热膨胀系数
// =============================================================================

/**
 * @brief 热膨胀应变系数: epsilon_th = alpha_T * (T - T_ref)
 * 
 * 单材料参数设计：每个实例仅存储一种材料的参数。
 * 多域场景通过 DomainMappedCoefficient 组合。
 */
class ThermalExpansionCoefficient : public Coefficient {
public:
    /// 构造时注入温度场引用和材料参数
    ThermalExpansionCoefficient(const GridFunction& T,
                                 Real alpha_T, Real T_ref = 293.15)
        : T_(&T), alpha_T_(alpha_T), T_ref_(T_ref) {}
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    const GridFunction* T_;
    Real alpha_T_;
    Real T_ref_;
};

// =============================================================================
// 向量系数
// =============================================================================

class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    virtual void eval(ElementTransform& trans, Real* result, Real t = 0.0) const = 0;
    virtual int dim() const = 0;
};

class ConstantVectorCoefficient : public VectorCoefficient {
public:
    explicit ConstantVectorCoefficient(Real x, Real y, Real z) : v_{x, y, z} {}
    void eval(ElementTransform&, Real* r, Real = 0.0) const override { 
        r[0] = v_[0]; r[1] = v_[1]; r[2] = v_[2]; 
    }
    int dim() const override { return 3; }
private:
    Real v_[3];
};

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP
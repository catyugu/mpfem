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

/// 分片常量系数（按域ID索引）
class PWConstCoefficient : public Coefficient {
public:
    explicit PWConstCoefficient(int numDomains = 0) : values_(numDomains, 0.0) {}
    explicit PWConstCoefficient(const std::vector<Real>& v) : values_(v) {}
    
    Real eval(ElementTransform& trans, Real = 0.0) const override;
    
    void set(int domainId, Real v) { 
        if (domainId >= 1 && domainId <= static_cast<int>(values_.size()))
            values_[domainId - 1] = v; 
    }
    Real get(int domainId) const { return values_[domainId - 1]; }
    void resize(int n, Real v = 0.0) { values_.resize(n, v); }
    int size() const { return static_cast<int>(values_.size()); }
    
private:
    std::vector<Real> values_;
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
 * 在构造时注入温度场引用，数据源唯一，无需手动更新。
 */
class TemperatureDependentConductivity : public Coefficient {
public:
    /// 构造时注入温度场引用
    TemperatureDependentConductivity(const GridFunction& T)
        : T_(&T) {}
    
    /// 设置温度依赖材料参数
    void setMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        ensureSize(domainId);
        rho0_[domainId - 1] = rho0;
        alpha_[domainId - 1] = alpha;
        tref_[domainId - 1] = tref;
    }
    
    /// 设置常量电导率（非温度依赖）
    void setConstantConductivity(int domainId, Real sigma) {
        ensureSize(domainId);
        sigma0_[domainId - 1] = sigma;
        rho0_[domainId - 1] = 0.0;
    }
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    void ensureSize(int domainId) {
        if (static_cast<int>(rho0_.size()) < domainId) {
            rho0_.resize(domainId, 0.0);
            alpha_.resize(domainId, 0.0);
            tref_.resize(domainId, 298.0);
            sigma0_.resize(domainId, 0.0);
        }
    }
    
    std::vector<Real> rho0_;
    std::vector<Real> alpha_;
    std::vector<Real> tref_;
    std::vector<Real> sigma0_;
    const GridFunction* T_ = nullptr;
};

// =============================================================================
// 焦耳热系数
// =============================================================================

/**
 * @brief 焦耳热系数: Q = sigma * |grad V|^2
 * 
 * 在构造时注入电势场和电导率引用，数据源唯一，无需手动更新。
 */
class JouleHeatCoefficient : public Coefficient {
public:
    /// 构造时注入电势场和电导率引用
    JouleHeatCoefficient(const GridFunction& V, const Coefficient& sigma,
                         std::set<int> domains = {})
        : V_(&V), sigma_(&sigma), domains_(std::move(domains)) {}
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    const GridFunction* V_ = nullptr;
    const Coefficient* sigma_ = nullptr;
    std::set<int> domains_;
};

// =============================================================================
// 热膨胀系数
// =============================================================================

/**
 * @brief 热膨胀应变系数: epsilon_th = alpha_T * (T - T_ref)
 * 
 * 在构造时注入温度场引用，数据源唯一，无需手动更新。
 */
class ThermalExpansionCoefficient : public Coefficient {
public:
    
    /// 构造时注入温度场引用
    explicit ThermalExpansionCoefficient(const GridFunction& T, Real T_ref = 293.15)
        : T_ref_(T_ref), T_(&T) {}
    
    /// 设置参考温度（默认293.15K = 20C）
    void setReferenceTemperature(Real T_ref) { T_ref_ = T_ref; }
    
    /// 添加域的热膨胀系数
    void setAlphaT(int domainId, Real alpha_T) {
        ensureSize(domainId);
        alpha_T_[domainId - 1] = alpha_T;
    }
    
    /// 获取热膨胀系数
    Real getAlphaT(int domainId) const {
        if (domainId < 1 || domainId > static_cast<int>(alpha_T_.size())) return 0.0;
        return alpha_T_[domainId - 1];
    }
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
    
private:
    void ensureSize(int domainId) {
        if (static_cast<int>(alpha_T_.size()) < domainId) {
            alpha_T_.resize(domainId, 0.0);
        }
    }
    
    std::vector<Real> alpha_T_;
    Real T_ref_ = 293.15;
    const GridFunction* T_ = nullptr;
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
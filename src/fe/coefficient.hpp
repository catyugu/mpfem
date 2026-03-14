#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <functional>
#include <vector>
#include <set>

namespace mpfem {

class ElementTransform;
class GridFunction;

// =============================================================================
// 标量系数基类
// =============================================================================

class Coefficient {
public:
    virtual ~Coefficient() = default;
    virtual Real eval(ElementTransform& trans) const = 0;
};

// =============================================================================
// 基础系数
// =============================================================================

/// 常量系数
class ConstantCoefficient : public Coefficient {
public:
    explicit ConstantCoefficient(Real c = 1.0) : value_(c) {}
    Real eval(ElementTransform&) const override { return value_; }
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
    
    Real eval(ElementTransform& trans) const override;
    
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
    using Func = std::function<Real(Real, Real, Real)>;
    explicit FunctionCoefficient(Func f) : func_(std::move(f)) {}
    Real eval(ElementTransform& trans) const override;
private:
    Func func_;
};

/// 网格函数系数（从解场获取值）
class GridFunctionCoefficient : public Coefficient {
public:
    explicit GridFunctionCoefficient(const GridFunction* gf = nullptr) : gf_(gf) {}
    Real eval(ElementTransform& trans) const override;
    void setField(const GridFunction* gf) { gf_ = gf; }
private:
    const GridFunction* gf_;
};

// =============================================================================
// 组合系数
// =============================================================================

/// 乘积系数: a * b
class ProductCoefficient : public Coefficient {
public:
    ProductCoefficient(const Coefficient* a, const Coefficient* b) : a_(a), b_(b) {}
    Real eval(ElementTransform& trans) const override;
private:
    const Coefficient* a_;
    const Coefficient* b_;
};

/// 缩放系数: scale * q
class ScaledCoefficient : public Coefficient {
public:
    ScaledCoefficient(const Coefficient* q, Real scale) : q_(q), scale_(scale) {}
    Real eval(ElementTransform& trans) const override;
private:
    const Coefficient* q_;
    Real scale_;
};

/// 域限制系数：仅在指定域上有效
class DomainRestrictedCoefficient : public Coefficient {
public:
    DomainRestrictedCoefficient(const Coefficient* q, const std::set<int>& domains)
        : q_(q), domains_(domains) {}
    
    Real eval(ElementTransform& trans) const override;
    
private:
    const Coefficient* q_;
    std::set<int> domains_;
};

// =============================================================================
// 温度依赖电导率系数（统一实现）
// =============================================================================

/**
 * @brief 温度依赖电导率：sigma = 1 / (rho0 * (1 + alpha * (T - Tref)))
 * 
 * 模型：
 *   rho = rho0 * (1 + alpha * (T - Tref))  线性电阻率模型
 *   sigma = 1 / rho
 * 
 * 如果 rho0 <= 0，则使用常量电导率 sigma0
 */
class TemperatureDependentConductivity : public Coefficient {
public:
    TemperatureDependentConductivity() = default;
    
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
        rho0_[domainId - 1] = 0.0;  // rho0 = 0 表示使用常量电导率
    }
    
    /// 设置温度场（非拥有指针）
    void setTemperatureField(const GridFunction* T) { T_ = T; }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    void ensureSize(int domainId) {
        if (static_cast<int>(rho0_.size()) < domainId) {
            rho0_.resize(domainId, 0.0);
            alpha_.resize(domainId, 0.0);
            tref_.resize(domainId, 298.0);
            sigma0_.resize(domainId, 0.0);
        }
    }
    
    std::vector<Real> rho0_;   ///< 参考温度下的电阻率
    std::vector<Real> alpha_;  ///< 温度系数
    std::vector<Real> tref_;   ///< 参考温度
    std::vector<Real> sigma0_; ///< 常量电导率
    const GridFunction* T_ = nullptr;
};

// =============================================================================
// 焦耳热系数
// =============================================================================

/**
 * @brief 焦耳热系数: Q = sigma * |grad V|^2
 */
class JouleHeatCoefficient : public Coefficient {
public:
    void setPotential(const GridFunction* V) { V_ = V; }
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    void setDomains(const std::set<int>& domains) { domains_ = domains; }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    const GridFunction* V_ = nullptr;
    const Coefficient* sigma_ = nullptr;
    std::set<int> domains_;
};

// =============================================================================
// 向量系数
// =============================================================================

class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    virtual void eval(ElementTransform& trans, Real* result) const = 0;
    virtual int dim() const = 0;
};

class ConstantVectorCoefficient : public VectorCoefficient {
public:
    explicit ConstantVectorCoefficient(Real x, Real y, Real z) : v_{x, y, z} {}
    void eval(ElementTransform&, Real* r) const override { r[0]=v_[0]; r[1]=v_[1]; r[2]=v_[2]; }
    int dim() const override { return 3; }
private:
    Real v_[3];
};

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP
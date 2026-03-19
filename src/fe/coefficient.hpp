#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <functional>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <variant>

namespace mpfem {

class ElementTransform;
class GridFunction;

// =============================================================================
// 系数类型标签
// =============================================================================

enum class CoefficientKind { Scalar, Vector, Matrix };

// =============================================================================
// 标量系数基类
// =============================================================================

class Coefficient {
public:
    virtual ~Coefficient() = default;
    virtual Real eval(ElementTransform& trans, Real t = 0.0) const = 0;
    static constexpr CoefficientKind kind = CoefficientKind::Scalar;
};

// =============================================================================
// 向量系数基类
// =============================================================================

class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    virtual void eval(ElementTransform& trans, Real* result, Real t = 0.0) const = 0;
    virtual int dim() const = 0;
    static constexpr CoefficientKind kind = CoefficientKind::Vector;
};

// =============================================================================
// 基础系数
// =============================================================================

class ConstantCoefficient : public Coefficient {
public:
    explicit ConstantCoefficient(Real c = 1.0) : value_(c) {}
    Real eval(ElementTransform&, Real = 0.0) const override { return value_; }
    void set(Real c) { value_ = c; }
    Real get() const { return value_; }
private:
    Real value_;
};

class FunctionCoefficient : public Coefficient {
public:
    using Func = std::function<Real(Real, Real, Real, Real)>;
    explicit FunctionCoefficient(Func f) : func_(std::move(f)) {}
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    Func func_;
};

// =============================================================================
// 域映射系数（模板化）
// =============================================================================

template<typename CoefType>
class DomainMappedCoefficientT;

template<>
class DomainMappedCoefficientT<Coefficient> : public Coefficient {
public:
    DomainMappedCoefficientT() = default;
    DomainMappedCoefficientT(DomainMappedCoefficientT&& o) noexcept
        : coefs_(std::move(o.coefs_)), defaultCoef_(o.defaultCoef_) { o.defaultCoef_ = nullptr; }
    DomainMappedCoefficientT& operator=(DomainMappedCoefficientT&& o) noexcept {
        if (this != &o) { coefs_ = std::move(o.coefs_); defaultCoef_ = o.defaultCoef_; o.defaultCoef_ = nullptr; }
        return *this;
    }
    
    void set(int domainId, const Coefficient* coef) { coefs_[domainId] = coef; }
    void set(const std::set<int>& domainIds, const Coefficient* coef) { for (int id : domainIds) coefs_[id] = coef; }
    void setAll(const Coefficient* coef) { defaultCoef_ = coef; coefs_.clear(); }
    const Coefficient* get(int domainId) const { auto it = coefs_.find(domainId); return it != coefs_.end() ? it->second : defaultCoef_; }
    bool empty() const { return coefs_.empty() && !defaultCoef_; }
    
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    std::map<int, const Coefficient*> coefs_;
    const Coefficient* defaultCoef_ = nullptr;
};

using DomainMappedCoefficient = DomainMappedCoefficientT<Coefficient>;
// =============================================================================
// 物理耦合系数
// =============================================================================

class TemperatureDependentConductivity : public Coefficient {
public:
    TemperatureDependentConductivity(const GridFunction& T, Real rho0, Real alpha, Real tref)
        : T_(&T), rho0_(rho0), alpha_(alpha), tref_(tref) {}
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    const GridFunction* T_;
    Real rho0_, alpha_, tref_;
};

class JouleHeatCoefficient : public Coefficient {
public:
    JouleHeatCoefficient(const GridFunction& V, const Coefficient& sigma) : V_(&V), sigma_(&sigma) {}
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    const GridFunction* V_;
    const Coefficient* sigma_;
};

class ThermalExpansionCoefficient : public Coefficient {
public:
    ThermalExpansionCoefficient(const GridFunction& T, Real alpha_T, Real T_ref = 293.15)
        : T_(&T), alpha_T_(alpha_T), T_ref_(T_ref) {}
    Real eval(ElementTransform& trans, Real t = 0.0) const override;
private:
    const GridFunction* T_;
    Real alpha_T_, T_ref_;
};

// =============================================================================
// 向量系数
// =============================================================================

class ConstantVectorCoefficient : public VectorCoefficient {
public:
    explicit ConstantVectorCoefficient(Real x, Real y, Real z) : v_{x, y, z} {}
    void eval(ElementTransform&, Real* r, Real = 0.0) const override { r[0] = v_[0]; r[1] = v_[1]; r[2] = v_[2]; }
    int dim() const override { return 3; }
private:
    Real v_[3];
};

// =============================================================================
// 类型擦除系数包装器
// =============================================================================

class AnyCoefficient {
public:
    AnyCoefficient() = default;
    explicit AnyCoefficient(std::unique_ptr<Coefficient> c) : data_(std::move(c)) {}
    explicit AnyCoefficient(std::unique_ptr<VectorCoefficient> c) : data_(std::move(c)) {}
    
    AnyCoefficient(AnyCoefficient&&) = default;
    AnyCoefficient& operator=(AnyCoefficient&&) = default;
    AnyCoefficient(const AnyCoefficient&) = delete;
    AnyCoefficient& operator=(const AnyCoefficient&) = delete;
    
    CoefficientKind kind() const {
        return data_.index() == 1 ? CoefficientKind::Scalar :
               data_.index() == 2 ? CoefficientKind::Vector : CoefficientKind::Scalar;
    }
    
    bool empty() const { return data_.index() == 0; }
    
    template<typename T>
    const T* get() const;
    
private:
    std::variant<std::monostate, std::unique_ptr<Coefficient>, std::unique_ptr<VectorCoefficient>> data_;
};

template<> inline const Coefficient* AnyCoefficient::get<Coefficient>() const {
    auto* p = std::get_if<1>(&data_);
    return p ? p->get() : nullptr;
}

template<> inline const VectorCoefficient* AnyCoefficient::get<VectorCoefficient>() const {
    auto* p = std::get_if<2>(&data_);
    return p ? p->get() : nullptr;
}

// 工厂函数
template<typename T, typename... Args>
AnyCoefficient makeCoefficient(Args&&... args) {
    if constexpr (std::is_base_of_v<Coefficient, T>)
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
    else if constexpr (std::is_base_of_v<VectorCoefficient, T>)
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP
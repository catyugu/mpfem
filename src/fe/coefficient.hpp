#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <functional>
#include <vector>
#include <map>

namespace mpfem {

class ElementTransform;
class GridFunction;

// =============================================================================
// Scalar Coefficient - 最小化接口
// =============================================================================

class Coefficient {
public:
    virtual ~Coefficient() = default;
    virtual Real eval(ElementTransform& trans) const = 0;
};

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

/// 分片常量系数（按域ID）
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

/// 场系数（非拥有指针）
class GridFunctionCoefficient : public Coefficient {
public:
    explicit GridFunctionCoefficient(const GridFunction* gf = nullptr) : gf_(gf) {}
    Real eval(ElementTransform& trans) const override;
    void setField(const GridFunction* gf) { gf_ = gf; }
private:
    const GridFunction* gf_;
};

/// 乘积系数
class ProductCoefficient : public Coefficient {
public:
    ProductCoefficient(const Coefficient* a, const Coefficient* b) : a_(a), b_(b) {}
    Real eval(ElementTransform& trans) const override { 
        return a_->eval(trans) * b_->eval(trans); 
    }
private:
    const Coefficient* a_;
    const Coefficient* b_;
};

/// 缩放系数
class ScaledCoefficient : public Coefficient {
public:
    ScaledCoefficient(const Coefficient* q, Real scale) : q_(q), scale_(scale) {}
    Real eval(ElementTransform& trans) const override { 
        return scale_ * q_->eval(trans); 
    }
private:
    const Coefficient* q_;
    Real scale_;
};

// =============================================================================
// 温度依赖系数（解耦设计：通过回调获取温度）
// =============================================================================

class TemperatureDependentCoefficient : public Coefficient {
public:
    using TempFunc = std::function<Real(int elemIdx, const Real* xi)>;
    
    void setTemperatureCallback(TempFunc func) { tempFunc_ = std::move(func); }
    
protected:
    TempFunc tempFunc_;
};

/// 温度依赖电导率
class TemperatureDependentConductivity : public TemperatureDependentCoefficient {
public:
    void setMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        rho0_[domainId] = rho0;
        alpha_[domainId] = alpha;
        tref_[domainId] = tref;
    }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    std::map<int, Real> rho0_;
    std::map<int, Real> alpha_;
    std::map<int, Real> tref_;
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
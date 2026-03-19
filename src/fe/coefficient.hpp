#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <functional>
#include <set>
#include <map>
#include <memory>
#include <variant>

namespace mpfem {

class ElementTransform;
class GridFunction;

// =============================================================================
// Coefficient type tags
// =============================================================================

enum class CoefficientKind { Scalar, Vector, Matrix };

// =============================================================================
// Base classes
// =============================================================================

class Coefficient {
public:
    virtual ~Coefficient() = default;
    virtual void eval(ElementTransform& trans, Real& result, Real t = 0.0) const = 0;
    static constexpr CoefficientKind kind = CoefficientKind::Scalar;
};

class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    virtual void eval(ElementTransform& trans, Vector3& result, Real t = 0.0) const = 0;
    static constexpr CoefficientKind kind = CoefficientKind::Vector;
};

class MatrixCoefficient {
public:
    virtual ~MatrixCoefficient() = default;
    virtual void eval(ElementTransform& trans, Matrix3& result, Real t = 0.0) const = 0;
    static constexpr CoefficientKind kind = CoefficientKind::Matrix;
};

// =============================================================================
// Lambda-based coefficients
// =============================================================================

class ScalarCoefficient : public Coefficient {
public:
    using Func = std::function<void(ElementTransform&, Real&, Real)>;
    explicit ScalarCoefficient(Func f) : func_(std::move(f)) {}
    void eval(ElementTransform& trans, Real& result, Real t) const override { func_(trans, result, t); }
private:
    Func func_;
};

class VectorFunctionCoefficient : public VectorCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Vector3&, Real)>;
    explicit VectorFunctionCoefficient(Func f) : func_(std::move(f)) {}
    void eval(ElementTransform& trans, Vector3& result, Real t) const override { func_(trans, result, t); }
private:
    Func func_;
};

class MatrixFunctionCoefficient : public MatrixCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Matrix3&, Real)>;
    explicit MatrixFunctionCoefficient(Func f) : func_(std::move(f)) {}
    void eval(ElementTransform& trans, Matrix3& result, Real t) const override { func_(trans, result, t); }
private:
    Func func_;
};

// =============================================================================
// DomainMappedCoefficient - Template implementation
// =============================================================================

// Type traits for result types
template<typename CoefBase> struct CoefTraits;
template<> struct CoefTraits<Coefficient> { using ResultType = Real; };
template<> struct CoefTraits<VectorCoefficient> { using ResultType = Vector3; };
template<> struct CoefTraits<MatrixCoefficient> { using ResultType = Matrix3; };

/**
 * @brief Domain-mapped coefficient supporting different coefficients per domain
 * 
 * Single template implementation handles scalar, vector, and matrix types.
 */
template<typename CoefBase>
class DomainMappedCoefficient : public CoefBase {
    using ResultType = typename CoefTraits<CoefBase>::ResultType;
    
public:
    DomainMappedCoefficient() = default;
    
    DomainMappedCoefficient(DomainMappedCoefficient&& o) noexcept
        : coefs_(std::move(o.coefs_)), defaultCoef_(o.defaultCoef_) { 
        o.defaultCoef_ = nullptr; 
    }
    
    DomainMappedCoefficient& operator=(DomainMappedCoefficient&& o) noexcept {
        if (this != &o) { 
            coefs_ = std::move(o.coefs_); 
            defaultCoef_ = o.defaultCoef_; 
            o.defaultCoef_ = nullptr; 
        }
        return *this;
    }
    
    void set(int domainId, const CoefBase* coef) { coefs_[domainId] = coef; }
    void set(const std::set<int>& domainIds, const CoefBase* coef) { 
        for (int id : domainIds) coefs_[id] = coef; 
    }
    void setAll(const CoefBase* coef) { defaultCoef_ = coef; coefs_.clear(); }
    
    const CoefBase* get(int domainId) const { 
        auto it = coefs_.find(domainId); 
        return it != coefs_.end() ? it->second : defaultCoef_; 
    }
    
    bool empty() const { return coefs_.empty() && !defaultCoef_; }
    
    void eval(ElementTransform& trans, ResultType& result, Real t) const override;

private:
    std::map<int, const CoefBase*> coefs_;
    const CoefBase* defaultCoef_ = nullptr;
};

// Type aliases
using DomainMappedScalarCoefficient = DomainMappedCoefficient<Coefficient>;
using DomainMappedVectorCoefficient = DomainMappedCoefficient<VectorCoefficient>;
using DomainMappedMatrixCoefficient = DomainMappedCoefficient<MatrixCoefficient>;

// =============================================================================
// Convenience functions for creating constant coefficients
// =============================================================================

inline std::unique_ptr<Coefficient> constantCoefficient(Real value) {
    return std::make_unique<ScalarCoefficient>(
        [value](ElementTransform&, Real& r, Real) { r = value; });
}

inline std::unique_ptr<VectorCoefficient> constantVectorCoefficient(Real x, Real y, Real z) {
    return std::make_unique<VectorFunctionCoefficient>(
        [x, y, z](ElementTransform&, Vector3& r, Real) { r << x, y, z; });
}

inline std::unique_ptr<MatrixCoefficient> diagonalMatrixCoefficient(Real diag) {
    return std::make_unique<MatrixFunctionCoefficient>(
        [diag](ElementTransform&, Matrix3& r, Real) { r = Matrix3::Identity() * diag; });
}

inline std::unique_ptr<MatrixCoefficient> constantMatrixCoefficient(const Matrix3& mat) {
    return std::make_unique<MatrixFunctionCoefficient>(
        [mat](ElementTransform&, Matrix3& r, Real) { r = mat; });
}

// =============================================================================
// AnyCoefficient - Type-erased wrapper
// =============================================================================

class AnyCoefficient {
public:
    AnyCoefficient() = default;
    explicit AnyCoefficient(std::unique_ptr<Coefficient> c) : data_(std::move(c)) {}
    explicit AnyCoefficient(std::unique_ptr<VectorCoefficient> c) : data_(std::move(c)) {}
    explicit AnyCoefficient(std::unique_ptr<MatrixCoefficient> c) : data_(std::move(c)) {}
    
    AnyCoefficient(AnyCoefficient&&) = default;
    AnyCoefficient& operator=(AnyCoefficient&&) = default;
    
    CoefficientKind kind() const {
        if (std::holds_alternative<std::unique_ptr<VectorCoefficient>>(data_)) return CoefficientKind::Vector;
        if (std::holds_alternative<std::unique_ptr<MatrixCoefficient>>(data_)) return CoefficientKind::Matrix;
        return CoefficientKind::Scalar;
    }
    
    bool empty() const { return std::holds_alternative<std::monostate>(data_); }
    
    template<typename T> const T* get() const;
    
private:
    std::variant<std::monostate, std::unique_ptr<Coefficient>, 
                 std::unique_ptr<VectorCoefficient>, std::unique_ptr<MatrixCoefficient>> data_;
};

template<> inline const Coefficient* AnyCoefficient::get<Coefficient>() const {
    auto* p = std::get_if<std::unique_ptr<Coefficient>>(&data_);
    return p ? p->get() : nullptr;
}

template<> inline const VectorCoefficient* AnyCoefficient::get<VectorCoefficient>() const {
    auto* p = std::get_if<std::unique_ptr<VectorCoefficient>>(&data_);
    return p ? p->get() : nullptr;
}

template<> inline const MatrixCoefficient* AnyCoefficient::get<MatrixCoefficient>() const {
    auto* p = std::get_if<std::unique_ptr<MatrixCoefficient>>(&data_);
    return p ? p->get() : nullptr;
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP
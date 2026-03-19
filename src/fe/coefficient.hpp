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
// Scalar coefficient base class
// =============================================================================

/**
 * @brief Scalar coefficient base class
 * 
 * Key design changes:
 * - eval() uses reference parameter for result: void eval(trans, Real& result, t)
 * - This enables static allocation and vectorization in integrators
 * - All specific coefficient types are replaced by lambda-based FunctionCoefficient
 */
class Coefficient {
public:
    virtual ~Coefficient() = default;
    
    /// Evaluate coefficient at integration point
    /// @param trans Element transform providing position and element info
    /// @param result Output: coefficient value
    /// @param t Time parameter (for transient problems)
    virtual void eval(ElementTransform& trans, Real& result, Real t = 0.0) const = 0;
    
    static constexpr CoefficientKind kind = CoefficientKind::Scalar;
};

// =============================================================================
// Vector coefficient base class
// =============================================================================

/**
 * @brief Vector coefficient base class (3D vectors)
 */
class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    
    /// Evaluate coefficient at integration point
    /// @param trans Element transform providing position and element info
    /// @param result Output: coefficient vector (3D)
    /// @param t Time parameter (for transient problems)
    virtual void eval(ElementTransform& trans, Vector3& result, Real t = 0.0) const = 0;
    
    static constexpr CoefficientKind kind = CoefficientKind::Vector;
};

// =============================================================================
// Matrix coefficient base class
// =============================================================================

/**
 * @brief Matrix coefficient base class (3x3 matrices)
 * 
 * Used for anisotropic material properties like:
 * - Anisotropic thermal conductivity tensor
 * - Anisotropic electrical conductivity tensor
 */
class MatrixCoefficient {
public:
    virtual ~MatrixCoefficient() = default;
    
    /// Evaluate coefficient at integration point
    /// @param trans Element transform providing position and element info
    /// @param result Output: coefficient matrix (3x3)
    /// @param t Time parameter (for transient problems)
    virtual void eval(ElementTransform& trans, Matrix3& result, Real t = 0.0) const = 0;
    
    static constexpr CoefficientKind kind = CoefficientKind::Matrix;
};

// =============================================================================
// FunctionCoefficient - Lambda-based coefficient (unified for all types)
// =============================================================================

/**
 * @brief Lambda-based scalar coefficient
 * 
 * Replaces all specific coefficient types (Constant, Function, etc.)
 * Example:
 *   auto coef = ScalarCoefficient([](ElementTransform& trans, Real& r, Real t) {
 *       r = 400.0;  // constant thermal conductivity
 *   });
 */
class ScalarCoefficient : public Coefficient {
public:
    using Func = std::function<void(ElementTransform&, Real&, Real)>;
    
    explicit ScalarCoefficient(Func f) : func_(std::move(f)) {}
    
    void eval(ElementTransform& trans, Real& result, Real t) const override {
        func_(trans, result, t);
    }
    
private:
    Func func_;
};

/**
 * @brief Lambda-based vector coefficient
 */
class VectorFunctionCoefficient : public VectorCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Vector3&, Real)>;
    
    explicit VectorFunctionCoefficient(Func f) : func_(std::move(f)) {}
    
    void eval(ElementTransform& trans, Vector3& result, Real t) const override {
        func_(trans, result, t);
    }
    
private:
    Func func_;
};

/**
 * @brief Lambda-based matrix coefficient
 */
class MatrixFunctionCoefficient : public MatrixCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Matrix3&, Real)>;
    
    explicit MatrixFunctionCoefficient(Func f) : func_(std::move(f)) {}
    
    void eval(ElementTransform& trans, Matrix3& result, Real t) const override {
        func_(trans, result, t);
    }
    
private:
    Func func_;
};

// =============================================================================
// Domain-mapped coefficient (template for all coefficient types)
// =============================================================================

/**
 * @brief Domain-mapped coefficient supporting different coefficients per domain
 * 
 * Template specializations for:
 * - DomainMappedCoefficient<Coefficient> : scalar per-domain
 * - DomainMappedCoefficient<VectorCoefficient> : vector per-domain
 * - DomainMappedCoefficient<MatrixCoefficient> : matrix per-domain
 */

// Primary template declaration (must be declared before specialization)
template<typename CoefType>
class DomainMappedCoefficient;

// Scalar version
template<>
class DomainMappedCoefficient<Coefficient> : public Coefficient {
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
    
    void set(int domainId, const Coefficient* coef) { coefs_[domainId] = coef; }
    void set(const std::set<int>& domainIds, const Coefficient* coef) { 
        for (int id : domainIds) coefs_[id] = coef; 
    }
    void setAll(const Coefficient* coef) { defaultCoef_ = coef; coefs_.clear(); }
    
    const Coefficient* get(int domainId) const { 
        auto it = coefs_.find(domainId); 
        return it != coefs_.end() ? it->second : defaultCoef_; 
    }
    
    bool empty() const { return coefs_.empty() && !defaultCoef_; }
    
    void eval(ElementTransform& trans, Real& result, Real t) const override;
    
private:
    std::map<int, const Coefficient*> coefs_;
    const Coefficient* defaultCoef_ = nullptr;
};

// Vector version
template<>
class DomainMappedCoefficient<VectorCoefficient> : public VectorCoefficient {
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
    
    void set(int domainId, const VectorCoefficient* coef) { coefs_[domainId] = coef; }
    void set(const std::set<int>& domainIds, const VectorCoefficient* coef) { 
        for (int id : domainIds) coefs_[id] = coef; 
    }
    void setAll(const VectorCoefficient* coef) { defaultCoef_ = coef; coefs_.clear(); }
    
    const VectorCoefficient* get(int domainId) const { 
        auto it = coefs_.find(domainId); 
        return it != coefs_.end() ? it->second : defaultCoef_; 
    }
    
    bool empty() const { return coefs_.empty() && !defaultCoef_; }
    
    void eval(ElementTransform& trans, Vector3& result, Real t) const override;
    
private:
    std::map<int, const VectorCoefficient*> coefs_;
    const VectorCoefficient* defaultCoef_ = nullptr;
};

// Matrix version
template<>
class DomainMappedCoefficient<MatrixCoefficient> : public MatrixCoefficient {
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
    
    void set(int domainId, const MatrixCoefficient* coef) { coefs_[domainId] = coef; }
    void set(const std::set<int>& domainIds, const MatrixCoefficient* coef) { 
        for (int id : domainIds) coefs_[id] = coef; 
    }
    void setAll(const MatrixCoefficient* coef) { defaultCoef_ = coef; coefs_.clear(); }
    
    const MatrixCoefficient* get(int domainId) const { 
        auto it = coefs_.find(domainId); 
        return it != coefs_.end() ? it->second : defaultCoef_; 
    }
    
    bool empty() const { return coefs_.empty() && !defaultCoef_; }
    
    void eval(ElementTransform& trans, Matrix3& result, Real t) const override;
    
private:
    std::map<int, const MatrixCoefficient*> coefs_;
    const MatrixCoefficient* defaultCoef_ = nullptr;
};

// Type aliases for convenience
using DomainMappedScalarCoefficient = DomainMappedCoefficient<Coefficient>;
using DomainMappedVectorCoefficient = DomainMappedCoefficient<VectorCoefficient>;
using DomainMappedMatrixCoefficient = DomainMappedCoefficient<MatrixCoefficient>;

// =============================================================================
// Constant coefficient helpers (convenience constructors)
// =============================================================================

/// Create a constant scalar coefficient
inline std::unique_ptr<Coefficient> constantCoefficient(Real value) {
    return std::make_unique<ScalarCoefficient>(
        [value](ElementTransform&, Real& r, Real) { r = value; });
}

/// Create a constant vector coefficient
inline std::unique_ptr<VectorCoefficient> constantVectorCoefficient(Real x, Real y, Real z) {
    return std::make_unique<VectorFunctionCoefficient>(
        [x, y, z](ElementTransform&, Vector3& r, Real) { 
            r = Vector3(x, y, z); 
        });
}

/// Create a constant matrix coefficient (diagonal)
inline std::unique_ptr<MatrixCoefficient> diagonalMatrixCoefficient(Real diag) {
    return std::make_unique<MatrixFunctionCoefficient>(
        [diag](ElementTransform&, Matrix3& r, Real) { 
            r = Matrix3::Identity() * diag; 
        });
}

/// Create a constant matrix coefficient (full)
inline std::unique_ptr<MatrixCoefficient> constantMatrixCoefficient(const Matrix3& mat) {
    return std::make_unique<MatrixFunctionCoefficient>(
        [mat](ElementTransform&, Matrix3& r, Real) { r = mat; });
}

// =============================================================================
// AnyCoefficient - Type-erased coefficient wrapper
// =============================================================================

/**
 * @brief Type-erased coefficient wrapper for storing coefficients of any type
 */
class AnyCoefficient {
public:
    AnyCoefficient() = default;
    
    explicit AnyCoefficient(std::unique_ptr<Coefficient> c) 
        : data_(std::move(c)) {}
    
    explicit AnyCoefficient(std::unique_ptr<VectorCoefficient> c) 
        : data_(std::move(c)) {}
    
    explicit AnyCoefficient(std::unique_ptr<MatrixCoefficient> c) 
        : data_(std::move(c)) {}
    
    AnyCoefficient(AnyCoefficient&&) = default;
    AnyCoefficient& operator=(AnyCoefficient&&) = default;
    AnyCoefficient(const AnyCoefficient&) = delete;
    AnyCoefficient& operator=(const AnyCoefficient&) = delete;
    
    CoefficientKind kind() const {
        if (std::holds_alternative<std::monostate>(data_)) return CoefficientKind::Scalar;
        if (std::holds_alternative<std::unique_ptr<Coefficient>>(data_)) return CoefficientKind::Scalar;
        if (std::holds_alternative<std::unique_ptr<VectorCoefficient>>(data_)) return CoefficientKind::Vector;
        return CoefficientKind::Matrix;
    }
    
    bool empty() const { 
        return std::holds_alternative<std::monostate>(data_); 
    }
    
    template<typename T>
    const T* get() const;
    
private:
    std::variant<std::monostate, 
                 std::unique_ptr<Coefficient>, 
                 std::unique_ptr<VectorCoefficient>,
                 std::unique_ptr<MatrixCoefficient>> data_;
};

// Template specializations for get()
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

// Factory functions
template<typename T, typename... Args>
AnyCoefficient makeCoefficient(Args&&... args) {
    if constexpr (std::is_base_of_v<Coefficient, T> && !std::is_base_of_v<VectorCoefficient, T> && !std::is_base_of_v<MatrixCoefficient, T>)
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
    else if constexpr (std::is_base_of_v<VectorCoefficient, T> && !std::is_base_of_v<MatrixCoefficient, T>)
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
    else if constexpr (std::is_base_of_v<MatrixCoefficient, T>)
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
    else
        return AnyCoefficient(std::make_unique<T>(std::forward<Args>(args)...));
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP

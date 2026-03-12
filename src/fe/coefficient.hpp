#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "fe/quadrature.hpp"
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <cmath>

namespace mpfem {

// Forward declarations
class ElementTransform;
class GridFunction;

/**
 * @file coefficient.hpp
 * @brief Coefficient classes for finite element assembly.
 * 
 * Design inspired by MFEM's coefficient system. Coefficients represent
 * spatially-varying (and optionally time-varying) functions used in
 * the assembly of bilinear and linear forms.
 * 
 * Types of coefficients:
 * - Coefficient: Scalar coefficient (returns Real)
 * - VectorCoefficient: Vector coefficient (returns Vector of given dimension)
 * - MatrixCoefficient: Matrix coefficient (returns Matrix of given dimensions)
 */

// =============================================================================
// Scalar Coefficient
// =============================================================================

/**
 * @brief Base class for scalar coefficients.
 * 
 * Coefficients are evaluated at integration points within elements.
 * The ElementTransform provides context about the element and spatial location.
 */
class Coefficient {
public:
    Coefficient() : time_(0.0) {}
    virtual ~Coefficient() = default;
    
    /// Set the time for time-dependent coefficients
    virtual void setTime(Real t) { time_ = t; }
    
    /// Get the current time
    Real time() const { return time_; }
    
    /**
     * @brief Evaluate the coefficient at an integration point.
     * @param trans Element transformation (contains element index and integration point).
     * @return Coefficient value.
     */
    virtual Real eval(ElementTransform& trans) const = 0;
    
    /**
     * @brief Evaluate at a specific integration point.
     * @param trans Element transformation.
     * @param ip Integration point.
     * @return Coefficient value.
     */
    virtual Real eval(ElementTransform& trans, const IntegrationPoint& ip) const;

protected:
    Real time_;
};

/**
 * @brief Constant scalar coefficient.
 */
class ConstantCoefficient : public Coefficient {
public:
    explicit ConstantCoefficient(Real c = 1.0) : constant_(c) {}
    
    Real eval(ElementTransform& /*trans*/) const override { return constant_; }
    
    /// Set the constant value
    void setValue(Real c) { constant_ = c; }
    
    /// Get the constant value
    Real value() const { return constant_; }
    
private:
    Real constant_;
};

/**
 * @brief Piecewise constant coefficient keyed by element attribute (domain ID).
 * 
 * For each element, the coefficient value is determined by the element's
 * attribute (domain ID). This is useful for material properties that vary
 * by region.
 */
class PWConstCoefficient : public Coefficient {
public:
    /// Construct with number of attributes
    explicit PWConstCoefficient(int numAttr = 0) : constants_(numAttr, 0.0) {}
    
    /// Construct from vector of constants (index = attribute - 1)
    explicit PWConstCoefficient(const std::vector<Real>& c) : constants_(c) {}
    
    Real eval(ElementTransform& trans) const override;
    
    /// Set constant for attribute i (1-indexed)
    void setConstant(int attr, Real c) {
        if (attr >= 1 && static_cast<size_t>(attr) <= constants_.size()) {
            constants_[attr - 1] = c;
        }
    }
    
    /// Get constant for attribute i (1-indexed)
    Real constant(int attr) const {
        if (attr < 1 || static_cast<size_t>(attr) > constants_.size()) {
            MPFEM_THROW(RangeException, 
                "PWConstCoefficient::constant: invalid attribute " + std::to_string(attr) +
                ", valid range is [1, " + std::to_string(constants_.size()) + "]");
        }
        return constants_[attr - 1];
    }
    
    /// Access operator (1-indexed)
    Real& operator()(int attr) {
        if (attr < 1) attr = 1;
        if (static_cast<size_t>(attr) > constants_.size()) {
            constants_.resize(attr, 0.0);
        }
        return constants_[attr - 1];
    }
    
    /// Get number of attributes
    int numAttributes() const { return static_cast<int>(constants_.size()); }
    
    /// Resize and initialize
    void resize(int numAttr, Real initValue = 0.0) {
        constants_.resize(numAttr, initValue);
    }
    
private:
    std::vector<Real> constants_;
};

/**
 * @brief Piecewise coefficient with arbitrary Coefficient objects per attribute.
 * 
 * Maps element attributes to Coefficient objects. Missing attributes
 * return zero.
 */
class PWCoefficient : public Coefficient {
public:
    PWCoefficient() = default;
    
    Real eval(ElementTransform& trans) const override;
    
    /// Set coefficient for an attribute (takes ownership if passed as unique_ptr)
    void setCoefficient(int attr, std::unique_ptr<Coefficient> coef) {
        pieces_[attr] = std::move(coef);
    }
    
    /// Set coefficient for an attribute (non-owning reference)
    void setCoefficientRef(int attr, Coefficient* coef) {
        pieceRefs_[attr] = coef;
    }
    
    /// Remove coefficient for an attribute
    void removeCoefficient(int attr) {
        pieces_.erase(attr);
        pieceRefs_.erase(attr);
    }
    
    /// Set time for all sub-coefficients
    void setTime(Real t) override {
        Coefficient::setTime(t);
        for (auto& [attr, coef] : pieces_) {
            coef->setTime(t);
        }
        for (auto& [attr, coef] : pieceRefs_) {
            coef->setTime(t);
        }
    }
    
private:
    std::map<int, std::unique_ptr<Coefficient>> pieces_;
    std::map<int, Coefficient*> pieceRefs_;  // Non-owning references
};

/**
 * @brief Function coefficient from std::function.
 * 
 * Wraps a C++ function or lambda that takes physical coordinates and time.
 */
class FunctionCoefficient : public Coefficient {
public:
    /// Time-independent function type: f(x, y, z)
    using FuncType = std::function<Real(Real, Real, Real)>;
    
    /// Time-dependent function type: f(x, y, z, t)
    using TDFuncType = std::function<Real(Real, Real, Real, Real)>;
    
    /// Construct from time-independent function
    explicit FunctionCoefficient(FuncType f) : func_(std::move(f)), tdFunc_(nullptr) {}
    
    /// Construct from time-dependent function
    explicit FunctionCoefficient(TDFuncType f, bool /*isTimeDependent*/) 
        : func_(nullptr), tdFunc_(std::move(f)) {}
    
    Real eval(ElementTransform& trans) const override;
    
private:
    FuncType func_;
    TDFuncType tdFunc_;
};

/**
 * @brief Coefficient defined by a GridFunction.
 * 
 * Evaluates a finite element field at integration points.
 * Optionally extracts a single component from a vector field.
 */
class GridFunctionCoefficient : public Coefficient {
public:
    GridFunctionCoefficient() : gf_(nullptr), component_(1) {}
    
    /// Construct from GridFunction (non-owning pointer)
    explicit GridFunctionCoefficient(const GridFunction* gf, int component = 1)
        : gf_(gf), component_(component) {}
    
    Real eval(ElementTransform& trans) const override;
    
    /// Set the GridFunction
    void setGridFunction(const GridFunction* gf) { gf_ = gf; }
    
    /// Get the GridFunction
    const GridFunction* gridFunction() const { return gf_; }
    
    /// Set component (for vector fields, 1-indexed)
    void setComponent(int comp) { component_ = comp; }
    
private:
    const GridFunction* gf_;
    int component_;
};

/**
 * @brief Coefficient that is zero outside specified attributes.
 */
class RestrictedCoefficient : public Coefficient {
public:
    RestrictedCoefficient() : coef_(nullptr) {}
    
    /// Construct from coefficient and active attributes
    RestrictedCoefficient(Coefficient& coef, const std::vector<int>& activeAttr)
        : coef_(&coef), activeAttr_(activeAttr.begin(), activeAttr.end()) {}
    
    Real eval(ElementTransform& trans) const override;
    
    /// Set the coefficient
    void setCoefficient(Coefficient& coef) { coef_ = &coef; }
    
    /// Set active attributes
    void setActiveAttributes(const std::vector<int>& attrs) {
        activeAttr_ = std::set<int>(attrs.begin(), attrs.end());
    }
    
    /// Add an active attribute
    void addActiveAttribute(int attr) { activeAttr_.insert(attr); }
    
    /// Set time
    void setTime(Real t) override {
        Coefficient::setTime(t);
        if (coef_) coef_->setTime(t);
    }
    
private:
    Coefficient* coef_;
    std::set<int> activeAttr_;
};

/**
 * @brief Product of two coefficients.
 */
class ProductCoefficient : public Coefficient {
public:
    ProductCoefficient(Coefficient* a, Coefficient* b) : a_(a), b_(b) {}
    
    Real eval(ElementTransform& trans) const override {
        return a_->eval(trans) * b_->eval(trans);
    }
    
    void setTime(Real t) override {
        Coefficient::setTime(t);
        a_->setTime(t);
        b_->setTime(t);
    }
    
private:
    Coefficient* a_;
    Coefficient* b_;
};

/**
 * @brief Ratio of two coefficients.
 */
class RatioCoefficient : public Coefficient {
public:
    RatioCoefficient(Coefficient* num, Coefficient* denom) 
        : num_(num), denom_(denom) {}
    
    Real eval(ElementTransform& trans) const override;
    
    void setTime(Real t) override {
        Coefficient::setTime(t);
        num_->setTime(t);
        denom_->setTime(t);
    }
    
private:
    Coefficient* num_;
    Coefficient* denom_;
};

/**
 * @brief Sum of two coefficients.
 */
class SumCoefficient : public Coefficient {
public:
    SumCoefficient(Coefficient* a, Coefficient* b, Real alpha = 1.0, Real beta = 1.0)
        : a_(a), b_(b), alpha_(alpha), beta_(beta) {}
    
    Real eval(ElementTransform& trans) const override {
        return alpha_ * a_->eval(trans) + beta_ * b_->eval(trans);
    }
    
    void setTime(Real t) override {
        Coefficient::setTime(t);
        a_->setTime(t);
        b_->setTime(t);
    }
    
private:
    Coefficient* a_;
    Coefficient* b_;
    Real alpha_;
    Real beta_;
};

/**
 * @brief Transformed coefficient: f(coef(x)).
 */
class TransformedCoefficient : public Coefficient {
public:
    using Transform1 = std::function<Real(Real)>;
    using Transform2 = std::function<Real(Real, Real)>;
    
    /// Single operand transform
    TransformedCoefficient(Coefficient* q, Transform1 f)
        : q1_(q), q2_(nullptr), transform1_(std::move(f)) {}
    
    /// Two operand transform
    TransformedCoefficient(Coefficient* q1, Coefficient* q2, Transform2 f)
        : q1_(q1), q2_(q2), transform2_(std::move(f)) {}
    
    Real eval(ElementTransform& trans) const override;
    
    void setTime(Real t) override {
        Coefficient::setTime(t);
        if (q1_) q1_->setTime(t);
        if (q2_) q2_->setTime(t);
    }
    
private:
    Coefficient* q1_;
    Coefficient* q2_;
    Transform1 transform1_;
    Transform2 transform2_;
};

// =============================================================================
// Temperature-Dependent Material Coefficients
// =============================================================================

/**
 * @brief Temperature-dependent electrical conductivity coefficient.
 * 
 * Implements linear resistivity model:
 *   rho(T) = rho0 * (1 + alpha * (T - Tref))
 *   sigma(T) = 1 / rho(T)
 */
class TemperatureDependentConductivityCoefficient : public Coefficient {
public:
    TemperatureDependentConductivityCoefficient() = default;
    
    /// Set material parameters per attribute
    void setMaterialFields(
        const std::vector<Real>& rho0,     ///< Reference resistivity at Tref
        const std::vector<Real>& alpha,    ///< Temperature coefficient
        const std::vector<Real>& tref,     ///< Reference temperature
        const std::vector<Real>& sigma0    ///< Constant conductivity (if rho0=0)
    );
    
    /// Set temperature field
    void setTemperatureField(const GridFunction* temperature) {
        temperature_ = temperature;
    }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    std::vector<Real> rho0_;
    std::vector<Real> alpha_;
    std::vector<Real> tref_;
    std::vector<Real> sigma0_;
    const GridFunction* temperature_ = nullptr;
};

/**
 * @brief Temperature-dependent thermal conductivity coefficient.
 */
class TemperatureDependentThermalConductivityCoefficient : public Coefficient {
public:
    TemperatureDependentThermalConductivityCoefficient() = default;
    
    /// Set material parameters per attribute
    void setMaterialFields(
        const std::vector<Real>& k0,       ///< Base thermal conductivity
        const std::vector<Real>& alpha,    ///< Temperature coefficient
        const std::vector<Real>& tref      ///< Reference temperature
    );
    
    /// Set temperature field
    void setTemperatureField(const GridFunction* temperature) {
        temperature_ = temperature;
    }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    std::vector<Real> k0_;
    std::vector<Real> alpha_;
    std::vector<Real> tref_;
    const GridFunction* temperature_ = nullptr;
};

// =============================================================================
// Vector Coefficient
// =============================================================================

/**
 * @brief Base class for vector coefficients.
 */
class VectorCoefficient {
public:
    explicit VectorCoefficient(int vdim = 3) : vdim_(vdim), time_(0.0) {}
    virtual ~VectorCoefficient() = default;
    
    /// Set the time
    virtual void setTime(Real t) { time_ = t; }
    
    /// Get the time
    Real time() const { return time_; }
    
    /// Get vector dimension
    int vdim() const { return vdim_; }
    
    /**
     * @brief Evaluate the vector coefficient.
     * @param trans Element transformation.
     * @param result Output vector (must have size >= vdim).
     */
    virtual void eval(ElementTransform& trans, Real* result) const = 0;
    
    /**
     * @brief Evaluate at a specific integration point.
     */
    virtual void eval(ElementTransform& trans, const IntegrationPoint& ip, 
                      Real* result) const;
    
protected:
    int vdim_;
    Real time_;
};

/**
 * @brief Constant vector coefficient.
 */
class VectorConstantCoefficient : public VectorCoefficient {
public:
    explicit VectorConstantCoefficient(const Vector3& v) 
        : VectorCoefficient(3), vec_(v) {}
    
    VectorConstantCoefficient(int vdim, const std::vector<Real>& v)
        : VectorCoefficient(vdim), values_(v) {}
    
    void eval(ElementTransform& /*trans*/, Real* result) const override {
        if (values_.empty()) {
            result[0] = vec_.x();
            result[1] = vec_.y();
            result[2] = vec_.z();
        } else {
            for (int i = 0; i < vdim_; ++i) {
                result[i] = values_[i];
            }
        }
    }
    
private:
    Vector3 vec_;
    std::vector<Real> values_;
};

/**
 * @brief Vector coefficient defined by a vector GridFunction.
 */
class VectorGridFunctionCoefficient : public VectorCoefficient {
public:
    VectorGridFunctionCoefficient() : VectorCoefficient(3), gf_(nullptr) {}
    
    explicit VectorGridFunctionCoefficient(const GridFunction* gf);
    
    void eval(ElementTransform& trans, Real* result) const override;
    
    void setGridFunction(const GridFunction* gf);
    
private:
    const GridFunction* gf_;
};

// =============================================================================
// Matrix Coefficient
// =============================================================================

/**
 * @brief Base class for matrix coefficients.
 * 
 * Used for anisotropic material properties (e.g., anisotropic conductivity
 * or thermal conductivity).
 */
class MatrixCoefficient {
public:
    MatrixCoefficient(int rows, int cols) 
        : rows_(rows), cols_(cols), time_(0.0) {}
    
    virtual ~MatrixCoefficient() = default;
    
    /// Set the time
    virtual void setTime(Real t) { time_ = t; }
    
    /// Get the time
    Real time() const { return time_; }
    
    /// Get number of rows
    int rows() const { return rows_; }
    
    /// Get number of columns
    int cols() const { return cols_; }
    
    /**
     * @brief Evaluate the matrix coefficient.
     * @param trans Element transformation.
     * @param result Output matrix (row-major, size = rows * cols).
     */
    virtual void eval(ElementTransform& trans, Real* result) const = 0;
    
    /**
     * @brief Evaluate as Eigen matrix.
     */
    Matrix evalMatrix(ElementTransform& trans) const {
        Matrix m(rows_, cols_);
        eval(trans, m.data());
        return m;
    }
    
protected:
    int rows_;
    int cols_;
    Real time_;
};

/**
 * @brief Identity matrix coefficient.
 */
class IdentityMatrixCoefficient : public MatrixCoefficient {
public:
    explicit IdentityMatrixCoefficient(int dim) : MatrixCoefficient(dim, dim) {}
    
    void eval(ElementTransform& /*trans*/, Real* result) const override {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i * cols_ + j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
};

/**
 * @brief Diagonal matrix coefficient (isotropic tensor).
 */
class DiagonalMatrixCoefficient : public MatrixCoefficient {
public:
    DiagonalMatrixCoefficient(int dim, Real value)
        : MatrixCoefficient(dim, dim), value_(value) {}
    
    void eval(ElementTransform& /*trans*/, Real* result) const override {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i * cols_ + j] = (i == j) ? value_ : 0.0;
            }
        }
    }
    
    void setValue(Real v) { value_ = v; }
    
private:
    Real value_;
};

/**
 * @brief Diagonal matrix from scalar coefficient.
 */
class DiagonalFromScalarCoefficient : public MatrixCoefficient {
public:
    DiagonalFromScalarCoefficient(int dim, Coefficient* scalar)
        : MatrixCoefficient(dim, dim), scalar_(scalar) {}
    
    void eval(ElementTransform& trans, Real* result) const override {
        Real v = scalar_->eval(trans);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result[i * cols_ + j] = (i == j) ? v : 0.0;
            }
        }
    }
    
    void setTime(Real t) override {
        MatrixCoefficient::setTime(t);
        scalar_->setTime(t);
    }
    
private:
    Coefficient* scalar_;
};

/**
 * @brief Constant matrix coefficient.
 */
class ConstantMatrixCoefficient : public MatrixCoefficient {
public:
    ConstantMatrixCoefficient(int rows, int cols, const std::vector<Real>& data)
        : MatrixCoefficient(rows, cols), data_(data) {}
    
    void eval(ElementTransform& /*trans*/, Real* result) const override {
        std::copy(data_.begin(), data_.end(), result);
    }
    
private:
    std::vector<Real> data_;
};

/**
 * @brief Piecewise constant matrix coefficient.
 */
class PWMatrixCoefficient : public MatrixCoefficient {
public:
    PWMatrixCoefficient(int rows, int cols)
        : MatrixCoefficient(rows, cols) {}
    
    void eval(ElementTransform& trans, Real* result) const override;
    
    /// Set matrix for an attribute
    void setMatrix(int attr, const std::vector<Real>& data) {
        matrices_[attr] = data;
    }
    
private:
    std::map<int, std::vector<Real>> matrices_;
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Create constant coefficient
inline std::unique_ptr<Coefficient> makeConstant(Real value) {
    return std::make_unique<ConstantCoefficient>(value);
}

/// Create piecewise constant coefficient
inline std::unique_ptr<Coefficient> makePWConst(const std::vector<Real>& values) {
    return std::make_unique<PWConstCoefficient>(values);
}

/// Create product coefficient
inline std::unique_ptr<Coefficient> makeProduct(Coefficient* a, Coefficient* b) {
    return std::make_unique<ProductCoefficient>(a, b);
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP

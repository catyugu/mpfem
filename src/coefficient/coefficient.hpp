#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "fe/fe_values.hpp"
#include "fe/element_transform.hpp"
#include "core/types.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Abstract base class for coefficients.
 * 
 * Coefficients provide spatially (and possibly temporally) varying values
 * that are used in integrators and boundary conditions. The unified interface
 * accepts time and FEValues (multi-field state) to support coupled physics.
 * 
 * Example usage:
 * @code
 * class ConductivityCoefficient : public Coefficient {
 *     Real eval(Real t, const FEValues& state, 
 *               Index elemIdx, const IntegrationPoint& ip,
 *               const ElementTransform& trans) const override {
 *         Real T = state.temperature(elemIdx, ip);
 *         return sigma0 / (1.0 + alpha * (T - Tref));
 *     }
 * };
 * @endcode
 */
class Coefficient {
public:
    virtual ~Coefficient() = default;

    /**
     * @brief Evaluate coefficient at an integration point.
     * @param t Current time.
     * @param state Multi-field state (for coupled physics).
     * @param elemIdx Element index.
     * @param ip Integration point.
     * @param trans Element transformation.
     * @return Coefficient value.
     */
    virtual Real eval(Real t, const FEValues& state,
                      Index elemIdx, const IntegrationPoint& ip,
                      const ElementTransform& trans) const = 0;

    /**
     * @brief Evaluate coefficient at reference coordinates.
     */
    virtual Real eval(Real t, const FEValues& state,
                      Index elemIdx, const Real* xi,
                      const ElementTransform& trans) const {
        IntegrationPoint ip(xi[0], xi[1], xi[2], 0.0);
        return eval(t, state, elemIdx, ip, trans);
    }
};

/**
 * @brief Coefficient that returns a constant value.
 */
class ConstantCoefficient : public Coefficient {
public:
    explicit ConstantCoefficient(Real value) : value_(value) {}

    Real eval(Real /*t*/, const FEValues& /*state*/,
              Index /*elemIdx*/, const IntegrationPoint& /*ip*/,
              const ElementTransform& /*trans*/) const override {
        return value_;
    }

    Real value() const { return value_; }
    void setValue(Real v) { value_ = v; }

private:
    Real value_;
};

/**
 * @brief Coefficient defined by a function.
 * 
 * The function signature is:
 * Real func(Real t, const Vector3& x, const FEValues& state)
 */
class FunctionCoefficient : public Coefficient {
public:
    using FuncType = std::function<Real(Real, const Vector3&, const FEValues&)>;

    explicit FunctionCoefficient(FuncType func) : func_(std::move(func)) {}

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override {
        Vector3 x;
        trans.transform(ip, x);
        return func_(t, x, state);
    }

private:
    FuncType func_;
};

/**
 * @brief Coefficient that depends on a field value.
 * 
 * Returns the value of another field at the integration point.
 */
class FieldCoefficient : public Coefficient {
public:
    explicit FieldCoefficient(FieldKind field) : field_(field) {}

    Real eval(Real /*t*/, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& /*trans*/) const override {
        return state.getValue(field_, elemIdx, ip);
    }

private:
    FieldKind field_;
};

/**
 * @brief Coefficient that returns the gradient magnitude of a field.
 */
class GradientMagnitudeCoefficient : public Coefficient {
public:
    explicit GradientMagnitudeCoefficient(FieldKind field) : field_(field) {}

    Real eval(Real /*t*/, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override {
        Vector3 grad = state.getGradient(field_, elemIdx, ip, trans);
        return grad.norm();
    }

private:
    FieldKind field_;
};

/**
 * @brief Coefficient that returns the squared gradient magnitude.
 * 
 * Useful for Joule heating: Q = sigma * |grad V|^2
 */
class GradientSquaredCoefficient : public Coefficient {
public:
    explicit GradientSquaredCoefficient(FieldKind field) : field_(field) {}

    Real eval(Real /*t*/, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override {
        Vector3 grad = state.getGradient(field_, elemIdx, ip, trans);
        return grad.squaredNorm();
    }

private:
    FieldKind field_;
};

// =============================================================================
// Helper functions for creating coefficients
// =============================================================================

/// Create a constant coefficient
inline std::unique_ptr<Coefficient> constCoeff(Real value) {
    return std::make_unique<ConstantCoefficient>(value);
}

/// Create a function coefficient
inline std::unique_ptr<Coefficient> funcCoeff(
    std::function<Real(Real, const Vector3&, const FEValues&)> func) {
    return std::make_unique<FunctionCoefficient>(std::move(func));
}

/// Create a field value coefficient
inline std::unique_ptr<Coefficient> fieldCoeff(FieldKind field) {
    return std::make_unique<FieldCoefficient>(field);
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP

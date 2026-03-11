#ifndef MPFEM_MATERIAL_COEFFICIENT_HPP
#define MPFEM_MATERIAL_COEFFICIENT_HPP

#include "coefficient/coefficient.hpp"
#include "model/material_database.hpp"
#include <memory>
#include <functional>

namespace mpfem {

/**
 * @brief Coefficient that returns material property based on element domain.
 * 
 * Uses the element's attribute (domain ID) to look up the material property.
 */
class MaterialCoefficient : public Coefficient {
public:
    /**
     * @brief Constructor.
     * @param materials Material database.
     * @param propName Property name (e.g., "electricconductivity").
     * @param domainToMaterial Map from domain ID to material tag.
     */
    MaterialCoefficient(const MaterialDatabase* materials,
                        const std::string& propName,
                        const std::map<int, std::string>& domainToMaterial);

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override;

private:
    const MaterialDatabase* materials_;
    std::string propName_;
    std::map<int, std::string> domainToMaterial_;
};

/**
 * @brief Temperature-dependent electric conductivity.
 * 
 * Uses linearized resistivity model:
 * sigma(T) = sigma0 / (1 + alpha * (T - Tref))
 * 
 * where sigma0 = 1/rho0 is the conductivity at reference temperature.
 */
class TemperatureDependentConductivity : public Coefficient {
public:
    /**
     * @brief Constructor.
     * @param materials Material database.
     * @param domainToMaterial Map from domain ID to material tag.
     */
    TemperatureDependentConductivity(const MaterialDatabase* materials,
                                     const std::map<int, std::string>& domainToMaterial);

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override;

private:
    const MaterialDatabase* materials_;
    std::map<int, std::string> domainToMaterial_;
};

/**
 * @brief Temperature-dependent coefficient using linear interpolation.
 * 
 * Uses the formula: value(T) = value0 * (1 + alpha * (T - Tref))
 */
class LinearTemperatureCoefficient : public Coefficient {
public:
    /**
     * @brief Constructor.
     * @param baseCoeff Base coefficient value at reference temperature.
     * @param alpha Temperature coefficient.
     * @param tref Reference temperature.
     */
    LinearTemperatureCoefficient(Real baseCoeff, Real alpha, Real tref = 293.15)
        : baseCoeff_(baseCoeff), alpha_(alpha), tref_(tref) {}

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override;

private:
    Real baseCoeff_;
    Real alpha_;
    Real tref_;
};

/**
 * @brief Product of two coefficients.
 */
class ProductCoefficient : public Coefficient {
public:
    ProductCoefficient(std::unique_ptr<Coefficient> a, std::unique_ptr<Coefficient> b)
        : a_(std::move(a)), b_(std::move(b)) {}

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override {
        return a_->eval(t, state, elemIdx, ip, trans) * 
               b_->eval(t, state, elemIdx, ip, trans);
    }

private:
    std::unique_ptr<Coefficient> a_;
    std::unique_ptr<Coefficient> b_;
};

/**
 * @brief Coefficient that multiplies another coefficient by a scalar.
 */
class ScaledCoefficient : public Coefficient {
public:
    ScaledCoefficient(Real scale, std::unique_ptr<Coefficient> coeff)
        : scale_(scale), coeff_(std::move(coeff)) {}

    Real eval(Real t, const FEValues& state,
              Index elemIdx, const IntegrationPoint& ip,
              const ElementTransform& trans) const override {
        return scale_ * coeff_->eval(t, state, elemIdx, ip, trans);
    }

private:
    Real scale_;
    std::unique_ptr<Coefficient> coeff_;
};

// =============================================================================
// Helper functions
// =============================================================================

/// Create product coefficient
inline std::unique_ptr<Coefficient> productCoeff(
    std::unique_ptr<Coefficient> a, std::unique_ptr<Coefficient> b) {
    return std::make_unique<ProductCoefficient>(std::move(a), std::move(b));
}

/// Create scaled coefficient
inline std::unique_ptr<Coefficient> scaleCoeff(
    Real scale, std::unique_ptr<Coefficient> coeff) {
    return std::make_unique<ScaledCoefficient>(scale, std::move(coeff));
}

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_COEFFICIENT_HPP

#include "coefficient/material_coefficient.hpp"
#include "fe/element_transform.hpp"
#include "mesh/mesh.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"

namespace mpfem {

// =============================================================================
// MaterialCoefficient
// =============================================================================

MaterialCoefficient::MaterialCoefficient(const MaterialDatabase* materials,
                                         const std::string& propName,
                                         const std::map<int, std::string>& domainToMaterial)
    : materials_(materials), propName_(propName), domainToMaterial_(domainToMaterial) {
    if (!materials_) {
        throw ArgumentException("MaterialDatabase is null in MaterialCoefficient");
    }
}

Real MaterialCoefficient::eval(Real /*t*/, const FEValues& /*state*/,
                               Index elemIdx, const IntegrationPoint& /*ip*/,
                               const ElementTransform& trans) const {
    // Get domain ID from element transform
    Index domainId = trans.elementAttribute();
    
    // Find material tag for this domain
    auto it = domainToMaterial_.find(static_cast<int>(domainId));
    if (it == domainToMaterial_.end()) {
        LOG_WARN("No material assigned for domain " << domainId);
        return 0.0;
    }
    
    // Get property value from material
    return materials_->getProperty(it->second, propName_);
}

// =============================================================================
// TemperatureDependentConductivity
// =============================================================================

TemperatureDependentConductivity::TemperatureDependentConductivity(
    const MaterialDatabase* materials,
    const std::map<int, std::string>& domainToMaterial)
    : materials_(materials), domainToMaterial_(domainToMaterial) {
    if (!materials_) {
        throw ArgumentException("MaterialDatabase is null in TemperatureDependentConductivity");
    }
}

Real TemperatureDependentConductivity::eval(Real /*t*/, const FEValues& state,
                                            Index elemIdx, const IntegrationPoint& ip,
                                            const ElementTransform& trans) const {
    // Get domain ID
    Index domainId = trans.elementAttribute();
    
    // Find material for this domain
    auto it = domainToMaterial_.find(static_cast<int>(domainId));
    if (it == domainToMaterial_.end()) {
        LOG_WARN("No material assigned for domain " << domainId);
        return 0.0;
    }
    
    const std::string& matTag = it->second;
    
    // Get temperature at this point
    Real T = state.hasField(FieldKind::Temperature) 
             ? state.temperature(elemIdx, ip) 
             : 293.15;  // Default to room temperature
    
    // Get material parameters
    Real rho0 = materials_->getProperty(matTag, "rho0", 1.72e-8);     // Reference resistivity
    Real alpha = materials_->getProperty(matTag, "alpha", 0.0039);    // Temperature coefficient
    Real Tref = materials_->getProperty(matTag, "Tref", 298.0);       // Reference temperature
    
    // Compute temperature-dependent conductivity
    // sigma(T) = 1 / (rho0 * (1 + alpha * (T - Tref)))
    Real rho = rho0 * (1.0 + alpha * (T - Tref));
    return 1.0 / rho;
}

// =============================================================================
// LinearTemperatureCoefficient
// =============================================================================

Real LinearTemperatureCoefficient::eval(Real /*t*/, const FEValues& state,
                                        Index elemIdx, const IntegrationPoint& ip,
                                        const ElementTransform& /*trans*/) const {
    Real T = state.hasField(FieldKind::Temperature) 
             ? state.temperature(elemIdx, ip) 
             : tref_;
    
    return baseCoeff_ * (1.0 + alpha_ * (T - tref_));
}

}  // namespace mpfem

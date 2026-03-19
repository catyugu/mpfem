#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include <string>
#include <map>
#include <vector>
#include <optional>

namespace mpfem {

/**
 * @brief Material property model.
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // General property storage
    std::map<std::string, double> properties;
    std::map<std::string, Matrix3> matrixProperties;  // Matrix (anisotropic) properties

    // Convenient accessors for common properties
    std::optional<double> youngModulus;           // E [Pa]
    std::optional<double> poissonRatio;           // nu [-]
    std::optional<double> density;                // rho [kg/m^3]
    std::optional<double> heatCapacity;           // Cp [J/(kg·K)]
    
    // Temperature-dependent resistivity: rho(T) = rho0 * (1 + alpha * (T - Tref))
    std::optional<double> rho0;
    std::optional<double> alpha;
    std::optional<double> tref;
    
    // Conductivities
    std::optional<double> electricConductivity;   // sigma [S/m] - isotropic
    std::optional<double> thermalConductivity;    // k [W/(m·K)] - isotropic
    std::optional<Matrix3> electricConductivityTensor;  // Anisotropic
    std::optional<Matrix3> thermalConductivityTensor;   // Anisotropic
    
    // Thermal expansion
    std::optional<double> thermalExpansion;       // alpha_T [1/K]

    std::optional<double> getProperty(const std::string& name) const {
        auto it = properties.find(name);
        return it != properties.end() ? std::optional<double>{it->second} : std::nullopt;
    }

    void setProperty(const std::string& name, double value) {
        properties[name] = value;
    }
    
    std::optional<Matrix3> getMatrixProperty(const std::string& name) const {
        auto it = matrixProperties.find(name);
        return it != matrixProperties.end() ? std::optional<Matrix3>{it->second} : std::nullopt;
    }
    
    void setMatrixProperty(const std::string& name, const Matrix3& mat) {
        matrixProperties[name] = mat;
    }
    
    /// Check if a property is anisotropic (has matrix form)
    bool isAnisotropic(const std::string& name) const {
        return matrixProperties.find(name) != matrixProperties.end();
    }

    std::optional<double> getElectricResistivity(double temperature) const {
        if (rho0.has_value() && rho0.value() > 0.0) {
            double a = alpha.value_or(0.0);
            double t = tref.value_or(298.0);
            return rho0.value() * (1.0 + a * (temperature - t));
        }
        if (electricConductivity.has_value() && electricConductivity.value() > 0.0) {
            return 1.0 / electricConductivity.value();
        }
        return std::nullopt;
    }

    std::optional<double> getElectricConductivity(double temperature) const {
        auto resistivity = getElectricResistivity(temperature);
        if (resistivity.has_value() && resistivity.value() > 0.0) {
            return 1.0 / resistivity.value();
        }
        return electricConductivity;
    }
    
    /// Get thermal conductivity (scalar for isotropic, or diagonal element)
    std::optional<double> getThermalConductivityScalar() const {
        if (thermalConductivity.has_value()) return thermalConductivity;
        if (thermalConductivityTensor.has_value()) {
            // Return average of diagonal for anisotropic
            const auto& m = thermalConductivityTensor.value();
            return (m(0,0) + m(1,1) + m(2,2)) / 3.0;
        }
        return std::nullopt;
    }
};

/**
 * @brief Database of material properties.
 */
class MaterialDatabase {
public:
    void addMaterial(const MaterialPropertyModel& material) {
        materials_[material.tag] = material;
    }

    const MaterialPropertyModel* getMaterial(const std::string& tag) const {
        auto it = materials_.find(tag);
        return it != materials_.end() ? &it->second : nullptr;
    }

    bool hasMaterial(const std::string& tag) const {
        return materials_.find(tag) != materials_.end();
    }

    std::vector<std::string> getMaterialTags() const {
        std::vector<std::string> tags;
        for (const auto& [tag, _] : materials_) tags.push_back(tag);
        return tags;
    }

    size_t size() const { return materials_.size(); }
    void clear() { materials_.clear(); }

    std::optional<double> getProperty(const std::string& tag, const std::string& propName) const {
        auto* mat = getMaterial(tag);
        if (!mat) return std::nullopt;
        return mat->getProperty(propName);
    }

    auto begin() const { return materials_.begin(); }
    auto end() const { return materials_.end(); }

private:
    std::map<std::string, MaterialPropertyModel> materials_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_DATABASE_HPP
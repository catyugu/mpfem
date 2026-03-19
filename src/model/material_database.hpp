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
 * 
 * Stores material properties for a single material.
 * All properties use std::optional to distinguish "not set" from "value is zero".
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // General property storage
    std::map<std::string, double> properties;

    // Convenient accessors for common properties (std::optional for unset detection)
    std::optional<double> youngModulus;           // E [Pa]
    std::optional<double> poissonRatio;           // nu [-]
    std::optional<double> density;                // rho [kg/m^3]
    std::optional<double> heatCapacity;           // Cp [J/(kg·K)]
    
    // Temperature-dependent resistivity: rho(T) = rho0 * (1 + alpha * (T - Tref))
    std::optional<double> rho0;                   // Reference resistivity [ohm·m]
    std::optional<double> alpha;                  // Temperature coefficient [1/K]
    std::optional<double> tref;                   // Reference temperature [K]
    
    // Conductivities (isotropic values extracted from tensor)
    std::optional<double> electricConductivity;   // sigma [S/m]
    std::optional<double> thermalConductivity;    // k [W/(m·K)]
    
    // Thermal expansion
    std::optional<double> thermalExpansion;       // alpha_T [1/K]

    /**
     * @brief Get a property by name.
     * @return Property value if found, nullopt otherwise.
     */
    std::optional<double> getProperty(const std::string& name) const {
        auto it = properties.find(name);
        return it != properties.end() ? std::optional<double>{it->second} : std::nullopt;
    }

    /**
     * @brief Set a property by name.
     */
    void setProperty(const std::string& name, double value) {
        properties[name] = value;
    }

    /**
     * @brief Compute temperature-dependent electric resistivity.
     * @param temperature Temperature in Kelvin.
     * @return Resistivity at given temperature, or nullopt if not defined.
     */
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

    /**
     * @brief Compute temperature-dependent electric conductivity.
     * @param temperature Temperature in Kelvin.
     * @return Conductivity at given temperature, or nullopt if not defined.
     */
    std::optional<double> getElectricConductivity(double temperature) const {
        auto resistivity = getElectricResistivity(temperature);
        if (resistivity.has_value() && resistivity.value() > 0.0) {
            return 1.0 / resistivity.value();
        }
        return electricConductivity;
    }
};

/**
 * @brief Database of material properties.
 * 
 * Stores materials indexed by their tag string.
 */
class MaterialDatabase {
public:
    /**
     * @brief Add a material to the database.
     */
    void addMaterial(const MaterialPropertyModel& material) {
        materials_[material.tag] = material;
    }

    /**
     * @brief Get a material by tag.
     * @return Pointer to material, or nullptr if not found.
     */
    const MaterialPropertyModel* getMaterial(const std::string& tag) const {
        auto it = materials_.find(tag);
        return it != materials_.end() ? &it->second : nullptr;
    }

    /**
     * @brief Check if a material exists.
     */
    bool hasMaterial(const std::string& tag) const {
        return materials_.find(tag) != materials_.end();
    }

    /**
     * @brief Get all material tags.
     */
    std::vector<std::string> getMaterialTags() const {
        std::vector<std::string> tags;
        for (const auto& [tag, _] : materials_) {
            tags.push_back(tag);
        }
        return tags;
    }

    /**
     * @brief Get number of materials.
     */
    size_t size() const { return materials_.size(); }

    /**
     * @brief Clear all materials.
     */
    void clear() { materials_.clear(); }

    /**
     * @brief Get a property from a material by tag and property name.
     * @param tag Material tag.
     * @param propName Property name.
     * @return Property value if found, nullopt otherwise.
     */
    std::optional<double> getProperty(const std::string& tag, const std::string& propName) const {
        auto* mat = getMaterial(tag);
        if (!mat) return std::nullopt;
        return mat->getProperty(propName);
    }

    // Iterator access
    auto begin() const { return materials_.begin(); }
    auto end() const { return materials_.end(); }

private:
    std::map<std::string, MaterialPropertyModel> materials_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_DATABASE_HPP

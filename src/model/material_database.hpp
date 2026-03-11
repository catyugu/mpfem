#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include <string>
#include <map>
#include <vector>

namespace mpfem {

/**
 * @brief Material property model.
 * 
 * Stores material properties for a single material.
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // General property storage
    std::map<std::string, double> properties;

    // Convenient accessors for common properties
    double youngModulus = 0.0;           // E [Pa]
    double poissonRatio = 0.0;           // nu [-]
    double density = 0.0;                // rho [kg/m^3]
    double heatCapacity = 0.0;           // Cp [J/(kg·K)]
    
    // Temperature-dependent resistivity: rho(T) = rho0 * (1 + alpha * (T - Tref))
    double rho0 = 0.0;                   // Reference resistivity [ohm·m]
    double alpha = 0.0;                  // Temperature coefficient [1/K]
    double tref = 293.15;                // Reference temperature [K]
    
    // Conductivities (isotropic values extracted from tensor)
    double electricConductivity = 0.0;   // sigma [S/m]
    double thermalConductivity = 0.0;    // k [W/(m·K)]
    
    // Thermal expansion
    double thermalExpansion = 0.0;       // alpha_T [1/K]

    /**
     * @brief Get a property by name.
     * @return Property value if found, 0.0 otherwise.
     */
    double getProperty(const std::string& name) const {
        auto it = properties.find(name);
        return it != properties.end() ? it->second : 0.0;
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
     * @return Resistivity at given temperature.
     */
    double getElectricResistivity(double temperature) const {
        if (rho0 <= 0.0) {
            return 1.0 / electricConductivity;
        }
        return rho0 * (1.0 + alpha * (temperature - tref));
    }

    /**
     * @brief Compute temperature-dependent electric conductivity.
     * @param temperature Temperature in Kelvin.
     * @return Conductivity at given temperature.
     */
    double getElectricConductivity(double temperature) const {
        double resistivity = getElectricResistivity(temperature);
        return resistivity > 0.0 ? 1.0 / resistivity : electricConductivity;
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

    // Iterator access
    auto begin() const { return materials_.begin(); }
    auto end() const { return materials_.end(); }

private:
    std::map<std::string, MaterialPropertyModel> materials_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_DATABASE_HPP

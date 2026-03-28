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
    std::map<std::string, Matrix3> matrixProperties;

    // Mechanical properties (scalar)
    std::optional<double> youngModulus;           // E [Pa]
    std::optional<double> poissonRatio;           // nu [-]
    std::optional<double> density;                // rho [kg/m^3]
    std::optional<double> heatCapacity;           // Cp [J/(kg·K)]
    
    // Temperature-dependent resistivity: rho(T) = rho0 * (1 + alpha * (T - Tref))
    std::optional<double> rho0;
    std::optional<double> alpha;
    std::optional<double> tref;
    
    // Conductivities (always matrix form)
    std::optional<Matrix3> electricConductivity;  // sigma tensor [S/m]
    std::optional<Matrix3> thermalConductivity;   // k tensor [W/(m·K)]
    
    // Thermal expansion
    std::optional<Matrix3> thermalExpansion;      // alpha_T tensor [1/K]

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
    
    /// Get temperature-dependent electric conductivity as matrix
    /// Returns diagonal matrix for isotropic case
    std::optional<Matrix3> getElectricConductivityMatrix(double temperature) const {
        // Temperature-dependent case
        if (rho0.has_value() && rho0.value() > 0.0) {
            double a = alpha.value_or(0.0);
            double t = tref.value_or(298.0);
            double resistivity = rho0.value() * (1.0 + a * (temperature - t));
            if (resistivity > 0.0) {
                return Matrix3::Identity() / resistivity;  // Diagonal matrix
            }
        }
        // Direct matrix form
        if (electricConductivity.has_value()) {
            return electricConductivity.value();
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

    auto begin() const { return materials_.begin(); }
    auto end() const { return materials_.end(); }

private:
    std::map<std::string, MaterialPropertyModel> materials_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_DATABASE_HPP

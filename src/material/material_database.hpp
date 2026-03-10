/**
 * @file material_database.hpp
 * @brief Material database for managing material definitions
 */

#ifndef MPFEM_MATERIAL_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_MATERIAL_DATABASE_HPP

#include "material_property.hpp"
#include "config/case_config.hpp"
#include <string>
#include <unordered_map>
#include <memory>

namespace mpfem {

/**
 * @brief Complete material definition
 */
class Material {
public:
    Material() = default;
    
    /**
     * @brief Construct from config
     */
    explicit Material(const MaterialConfig& config);
    
    // Move constructor
    Material(Material&& other) noexcept = default;
    
    // Move assignment
    Material& operator=(Material&& other) noexcept = default;
    
    // Copy operations deleted (contains unique_ptr)
    Material(const Material&) = delete;
    Material& operator=(const Material&) = delete;
    
    /// Get material tag
    const std::string& tag() const { return tag_; }
    
    /// Get material label
    const std::string& label() const { return label_; }
    
    /**
     * @brief Get a property by name
     * @param name Property name (e.g., "electricconductivity", "thermalconductivity")
     * @return Property or nullptr if not found
     */
    const Property* get_property(const std::string& name) const {
        auto it = properties_.find(name);
        if (it != properties_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Get scalar property value
     * @param name Property name
     * @param evaluator Material evaluator for field-dependent properties
     * @return Property value, or 0 if not found
     */
    Scalar get_scalar(const std::string& name, 
                      const MaterialEvaluator& evaluator) const;
    
    /**
     * @brief Get conductivity (handles isotropic and anisotropic)
     * @param evaluator Material evaluator
     * @return Conductivity tensor
     */
    Tensor<2, 3> get_conductivity(const MaterialEvaluator& evaluator) const;
    
    /**
     * @brief Get thermal conductivity
     */
    Tensor<2, 3> get_thermal_conductivity(const MaterialEvaluator& evaluator) const;
    
    /**
     * @brief Get Young's modulus
     */
    Scalar get_youngs_modulus() const;
    
    /**
     * @brief Get Poisson's ratio
     */
    Scalar get_poissons_ratio() const;
    
    /**
     * @brief Get thermal expansion coefficient
     */
    Tensor<2, 3> get_thermal_expansion() const;
    
    /**
     * @brief Get density
     */
    Scalar get_density() const;
    
    /**
     * @brief Get heat capacity
     */
    Scalar get_heat_capacity() const;
    
    /**
     * @brief Set a property
     */
    void set_property(const std::string& name, Property prop) {
        properties_[name] = std::move(prop);
    }
    
    /**
     * @brief Check if property exists
     */
    bool has_property(const std::string& name) const {
        return properties_.find(name) != properties_.end();
    }
    
    /**
     * @brief Get field dependencies
     */
    const std::vector<std::string>& field_dependencies() const {
        return field_dependencies_;
    }
    
private:
    std::string tag_;
    std::string label_;
    std::unordered_map<std::string, Property> properties_;
    std::vector<std::string> field_dependencies_;
    
    // Special properties for quick access
    std::unique_ptr<LinearizedResistivity> linearized_resistivity_;
    
    void parse_properties_from_config(const MaterialConfig& config);
};

/**
 * @brief Material database managing all materials
 */
class MaterialDB {
public:
    /**
     * @brief Build database from parsed configuration
     */
    void build(const MaterialDatabase& config_db);
    
    /**
     * @brief Get material by tag
     */
    const Material* get(const std::string& tag) const {
        auto it = materials_.find(tag);
        if (it != materials_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Add a material
     */
    void add(const std::string& tag, Material mat) {
        materials_[tag] = std::move(mat);
    }
    
    /**
     * @brief Get number of materials
     */
    size_t size() const { return materials_.size(); }
    
    /**
     * @brief Clear all materials
     */
    void clear() { materials_.clear(); }
    
private:
    std::unordered_map<std::string, Material> materials_;
};

/**
 * @brief Material evaluator - provides field values at quadrature points
 * 
 * This class bridges the gap between global solution vectors and
 * material property evaluation. It provides field values at the
 * current quadrature point.
 */
class MaterialEvaluator {
public:
    MaterialEvaluator() = default;
    
    /**
     * @brief Set field value for current quadrature point
     */
    void set_field(const std::string& name, Scalar value) {
        field_values_[name] = value;
    }
    
    /**
     * @brief Set temperature value (convenience method)
     */
    void set_temperature(Scalar T) {
        field_values_["temperature"] = T;
    }
    
    /**
     * @brief Get field value
     */
    Scalar get_field(const std::string& name) const {
        auto it = field_values_.find(name);
        if (it != field_values_.end()) {
            return it->second;
        }
        return 0.0;
    }
    
    /**
     * @brief Get temperature
     */
    Scalar temperature() const {
        return get_field("temperature");
    }
    
    /**
     * @brief Check if field is set
     */
    bool has_field(const std::string& name) const {
        return field_values_.find(name) != field_values_.end();
    }
    
    /**
     * @brief Clear all field values
     */
    void clear() {
        field_values_.clear();
    }
    
private:
    std::unordered_map<std::string, Scalar> field_values_;
};

// Implementation of FieldDependentProperty::value
inline Scalar FieldDependentProperty::value(const MaterialEvaluator& evaluator) const {
    Scalar field_value = evaluator.get_field(field_name_);
    return func_(field_value);
}

// Implementation of LinearizedResistivity::value
inline Scalar LinearizedResistivity::value(const MaterialEvaluator& evaluator) const {
    Scalar T = evaluator.temperature();
    return at_temperature(T);
}

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_MATERIAL_DATABASE_HPP

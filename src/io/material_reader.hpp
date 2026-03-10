/**
 * @file material_reader.hpp
 * @brief Material property reader from XML files
 */

#ifndef MPFEM_IO_MATERIAL_READER_HPP
#define MPFEM_IO_MATERIAL_READER_HPP

#include "variable_system.hpp"
#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <optional>

namespace tinyxml2 {
    class XMLElement;
    class XMLDocument;
}

namespace mpfem {

/**
 * @brief Material property value - can be scalar or tensor
 */
struct PropertyValue {
    std::string name;                   ///< Property name
    std::string raw_value;              ///< Raw value string
    
    bool is_tensor = false;             ///< Whether this is a tensor
    Scalar scalar_value = 0.0;          ///< Scalar value (SI units)
    Tensor<2, 3> tensor_value;          ///< Tensor value (SI units)
    
    std::string unit;                   ///< Original unit
    
    bool valid = false;                 ///< Whether parsing succeeded
};

/**
 * @brief Property group (e.g., "Enu" for elastic properties)
 */
struct PropertyGroup {
    std::string tag;                    ///< Group tag
    std::string description;            ///< Group description
    std::vector<PropertyValue> properties;
    
    // For temperature-dependent properties
    bool temperature_dependent = false;
    std::set<std::string> dependencies; ///< e.g., {"temperature"}
};

/**
 * @brief Complete material definition
 */
struct Material {
    std::string tag;                    ///< Material tag (mat1, mat2, etc.)
    std::string label;                  ///< Material label (Copper, etc.)
    std::string family;                 ///< Material family
    
    std::vector<PropertyGroup> groups;
    std::map<std::string, PropertyValue> properties;  ///< Quick lookup by name
    
    /**
     * @brief Get a scalar property by name
     */
    std::optional<Scalar> get_scalar(const std::string& name) const;
    
    /**
     * @brief Get a tensor property by name
     */
    std::optional<Tensor<2, 3>> get_tensor(const std::string& name) const;
    
    /**
     * @brief Check if property exists
     */
    bool has_property(const std::string& name) const;
    
    /**
     * @brief Check if material has temperature-dependent properties
     */
    bool is_temperature_dependent() const;
    
    /**
     * @brief Get temperature-dependent resistivity parameters
     * @return tuple (rho0, alpha, Tref) if available
     */
    std::optional<std::tuple<Scalar, Scalar, Scalar>> get_resistivity_params() const;
};

/**
 * @brief Material reader from XML files
 * 
 * Reads material.xml files in COMSOL format.
 */
class MaterialReader {
public:
    MaterialReader();
    ~MaterialReader();
    
    /**
     * @brief Read materials from XML file
     * @param filename Path to material.xml
     * @return Map of material tag to Material
     */
    std::map<std::string, Material> read(const std::string& filename);
    
    /**
     * @brief Get last error message
     */
    const std::string& error() const { return error_; }

private:
    std::unique_ptr<tinyxml2::XMLDocument> doc_;
    std::string error_;
    VariableSystem var_system_;  ///< For unit conversion
    
    /**
     * @brief Parse a single material element
     */
    Material parse_material(tinyxml2::XMLElement* elem);
    
    /**
     * @brief Parse a property group
     */
    PropertyGroup parse_property_group(tinyxml2::XMLElement* elem);
    
    /**
     * @brief Parse a property value
     */
    PropertyValue parse_property_value(const std::string& name, 
                                        const std::string& value_str);
    
    /**
     * @brief Parse tensor value like "{'400[W/(m*K)]','0',...}"
     */
    bool parse_tensor_value(const std::string& str, Tensor<2, 3>& tensor);
    
    /**
     * @brief Parse scalar value with optional unit
     */
    bool parse_scalar_value(const std::string& str, Scalar& value, std::string& unit);
};

} // namespace mpfem

#endif // MPFEM_IO_MATERIAL_READER_HPP

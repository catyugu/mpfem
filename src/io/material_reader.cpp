/**
 * @file material_reader.cpp
 * @brief Implementation of material property reader
 */

#include "material_reader.hpp"
#include "core/logger.hpp"

#include <tinyxml2.h>
#include <sstream>
#include <regex>
#include <algorithm>

namespace mpfem {

// ============================================================
// Material Implementation
// ============================================================

std::optional<Scalar> Material::get_scalar(const std::string& name) const {
    auto it = properties.find(name);
    if (it == properties.end() || it->second.is_tensor) {
        return std::nullopt;
    }
    return it->second.scalar_value;
}

std::optional<Tensor<2, 3>> Material::get_tensor(const std::string& name) const {
    auto it = properties.find(name);
    if (it == properties.end() || !it->second.is_tensor) {
        return std::nullopt;
    }
    return it->second.tensor_value;
}

bool Material::has_property(const std::string& name) const {
    return properties.find(name) != properties.end();
}

bool Material::is_temperature_dependent() const {
    for (const auto& group : groups) {
        if (group.temperature_dependent) {
            return true;
        }
    }
    return false;
}

std::optional<std::tuple<Scalar, Scalar, Scalar>> Material::get_resistivity_params() const {
    // Look for linearized resistivity parameters
    auto rho0 = get_scalar("rho0");
    auto alpha = get_scalar("alpha");
    auto Tref = get_scalar("Tref");
    
    if (rho0 && alpha && Tref) {
        return std::make_tuple(*rho0, *alpha, *Tref);
    }
    return std::nullopt;
}

// ============================================================
// MaterialReader Implementation
// ============================================================

MaterialReader::MaterialReader()
    : doc_(std::make_unique<tinyxml2::XMLDocument>()) {
}

MaterialReader::~MaterialReader() = default;

std::map<std::string, Material> MaterialReader::read(const std::string& filename) {
    std::map<std::string, Material> materials;
    
    tinyxml2::XMLError err = doc_->LoadFile(filename.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        error_ = "Failed to load material file: " + filename;
        MPFEM_ERROR(error_);
        return materials;
    }
    
    tinyxml2::XMLElement* archive = doc_->RootElement();
    if (!archive) {
        error_ = "Empty material document";
        MPFEM_ERROR(error_);
        return materials;
    }
    
    tinyxml2::XMLElement* model = archive->FirstChildElement("model");
    if (!model) {
        error_ = "No model element found";
        MPFEM_ERROR(error_);
        return materials;
    }
    
    // Parse all materials
    for (tinyxml2::XMLElement* mat_elem = model->FirstChildElement("material");
         mat_elem != nullptr;
         mat_elem = mat_elem->NextSiblingElement("material")) {
        
        Material mat = parse_material(mat_elem);
        if (!mat.tag.empty()) {
            materials[mat.tag] = std::move(mat);
        }
    }
    
    MPFEM_INFO("Loaded " << materials.size() << " materials");
    
    return materials;
}

Material MaterialReader::parse_material(tinyxml2::XMLElement* elem) {
    Material mat;
    
    // Get tag
    const char* tag = elem->Attribute("tag");
    if (tag) {
        mat.tag = tag;
    }
    
    // Get label
    tinyxml2::XMLElement* label_elem = elem->FirstChildElement("label");
    if (label_elem) {
        const char* label = label_elem->Attribute("label");
        if (label) {
            mat.label = label;
        }
    }
    
    // Get family
    for (tinyxml2::XMLElement* set_elem = elem->FirstChildElement("set");
         set_elem != nullptr;
         set_elem = set_elem->NextSiblingElement("set")) {
        
        const char* name = set_elem->Attribute("name");
        const char* value = set_elem->Attribute("value");
        
        if (name && std::string(name) == "family" && value) {
            mat.family = value;
            break;
        }
    }
    
    // Parse property groups
    for (tinyxml2::XMLElement* group_elem = elem->FirstChildElement("propertyGroup");
         group_elem != nullptr;
         group_elem = group_elem->NextSiblingElement("propertyGroup")) {
        
        PropertyGroup group = parse_property_group(group_elem);
        mat.groups.push_back(std::move(group));
    }
    
    // Build quick lookup map
    for (const auto& group : mat.groups) {
        for (const auto& prop : group.properties) {
            mat.properties[prop.name] = prop;
        }
    }
    
    MPFEM_INFO("Parsed material: " << mat.tag << " (" << mat.label << ")");
    MPFEM_INFO("  - " << mat.properties.size() << " properties");
    if (mat.is_temperature_dependent()) {
        MPFEM_INFO("  - Temperature dependent");
    }
    
    return mat;
}

PropertyGroup MaterialReader::parse_property_group(tinyxml2::XMLElement* elem) {
    PropertyGroup group;
    
    // Get tag
    const char* tag = elem->Attribute("tag");
    if (tag) {
        group.tag = tag;
    }
    
    // Get description
    const char* descr = elem->Attribute("descr");
    if (descr) {
        group.description = descr;
    }
    
    // Check for temperature dependency
    for (tinyxml2::XMLElement* input = elem->FirstChildElement("addInput");
         input != nullptr;
         input = input->NextSiblingElement("addInput")) {
        
        const char* quantity = input->Attribute("quantity");
        if (quantity) {
            group.temperature_dependent = true;
            group.dependencies.insert(quantity);
        }
    }
    
    // Parse properties
    for (tinyxml2::XMLElement* set_elem = elem->FirstChildElement("set");
         set_elem != nullptr;
         set_elem = set_elem->NextSiblingElement("set")) {
        
        const char* name = set_elem->Attribute("name");
        const char* value = set_elem->Attribute("value");
        
        if (name && value) {
            PropertyValue prop = parse_property_value(name, value);
            
            // Mark as temperature dependent if group is
            if (group.temperature_dependent) {
                // These properties depend on temperature
            }
            
            group.properties.push_back(std::move(prop));
        }
    }
    
    return group;
}

PropertyValue MaterialReader::parse_property_value(const std::string& name,
                                                    const std::string& value_str) {
    PropertyValue prop;
    prop.name = name;
    prop.raw_value = value_str;
    
    // Check if it's a tensor (starts with '{')
    if (!value_str.empty() && value_str[0] == '{') {
        prop.is_tensor = true;
        prop.valid = parse_tensor_value(value_str, prop.tensor_value);
        
        if (prop.valid) {
            // Extract diagonal for scalar representation
            prop.scalar_value = prop.tensor_value(0, 0);
        }
    } else {
        prop.is_tensor = false;
        prop.valid = parse_scalar_value(value_str, prop.scalar_value, prop.unit);
    }
    
    if (!prop.valid) {
        MPFEM_WARN("Failed to parse property: " << name << " = " << value_str);
    }
    
    return prop;
}

bool MaterialReader::parse_tensor_value(const std::string& str, Tensor<2, 3>& tensor) {
    // Format: "{'400[W/(m*K)]','0','0','0','400[W/(m*K)]','0','0','0','400[W/(m*K)]'}"
    // or simpler: "{'400','0',...}"
    
    tensor = Tensor<2, 3>::Zero();
    
    // Remove outer braces
    std::string content = str;
    if (content.front() == '{') content.erase(0, 1);
    if (content.back() == '}') content.pop_back();
    
    // Split by comma, but be careful with nested braces/quotes
    std::vector<std::string> elements;
    std::string current;
    bool in_quote = false;
    
    for (char c : content) {
        if (c == '\'') {
            in_quote = !in_quote;
        } else if (c == ',' && !in_quote) {
            elements.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        elements.push_back(current);
    }
    
    // We expect 9 elements for a 3x3 tensor
    if (elements.size() != 9) {
        MPFEM_WARN("Invalid tensor size: expected 9 elements, got " << elements.size());
        return false;
    }
    
    // Parse each element
    for (int i = 0; i < 9; ++i) {
        std::string elem = elements[i];
        
        // Remove quotes
        if (!elem.empty() && elem.front() == '\'') elem.erase(0, 1);
        if (!elem.empty() && elem.back() == '\'') elem.pop_back();
        
        Scalar value = 0;
        std::string unit;
        if (!parse_scalar_value(elem, value, unit)) {
            // Try parsing as plain number
            try {
                value = std::stod(elem);
            } catch (...) {
                MPFEM_WARN("Failed to parse tensor element: " << elem);
                return false;
            }
        }
        
        int row = i / 3;
        int col = i % 3;
        tensor(row, col) = value;
    }
    
    return true;
}

bool MaterialReader::parse_scalar_value(const std::string& str, Scalar& value, 
                                         std::string& unit) {
    // Use VariableSystem for parsing
    ParsedValue parsed = var_system_.parse_value(str);
    
    if (parsed.valid) {
        value = parsed.value;
        unit = parsed.unit;
        return true;
    }
    
    // Try as plain number
    try {
        size_t idx;
        value = std::stod(str, &idx);
        if (idx == str.length()) {
            unit = "";
            return true;
        }
    } catch (...) {
        // Not a plain number
    }
    
    return false;
}

} // namespace mpfem

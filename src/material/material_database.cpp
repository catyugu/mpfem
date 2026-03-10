/**
 * @file material_database.cpp
 * @brief Material database implementation
 */

#include "material_database.hpp"
#include "config/case_parser.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

Material::Material(const MaterialConfig& config)
    : tag_(config.tag), label_(config.label) {
    parse_properties_from_config(config);
    field_dependencies_ = config.inputs;
}

void Material::parse_properties_from_config(const MaterialConfig& config) {
    // Parse global properties
    for (const auto& prop : config.properties) {
        if (prop.is_tensor) {
            auto values = CaseParser::parse_tensor(prop.value);
            properties_[prop.name] = Property(values);
        } else {
            Scalar value = CaseParser::parse_value_with_unit(prop.value);
            properties_[prop.name] = Property(value);
        }
    }
    
    // Parse property groups
    for (const auto& [group_tag, props] : config.property_groups) {
        if (group_tag == "Enu") {
            // Young's modulus and Poisson's ratio
            for (const auto& prop : props) {
                Scalar value = CaseParser::parse_value_with_unit(prop.value);
                if (prop.name == "E") {
                    properties_["youngs_modulus"] = Property(value);
                } else if (prop.name == "nu") {
                    properties_["poisson_ratio"] = Property(value);
                }
            }
        } else if (group_tag == "linzRes") {
            // Linearized resistivity
            Scalar rho0 = 0, alpha = 0, Tref = 298;
            for (const auto& prop : props) {
                Scalar value = CaseParser::parse_value_with_unit(prop.value);
                if (prop.name == "rho0") {
                    rho0 = value;
                } else if (prop.name == "alpha") {
                    alpha = value;
                } else if (prop.name == "Tref") {
                    Tref = value;
                }
            }
            linearized_resistivity_ = std::make_unique<LinearizedResistivity>(rho0, alpha, Tref);
            field_dependencies_.push_back("temperature");
        } else if (group_tag == "def") {
            // Default properties
            for (const auto& prop : props) {
                if (prop.is_tensor) {
                    auto values = CaseParser::parse_tensor(prop.value);
                    properties_[prop.name] = Property(values);
                } else {
                    Scalar value = CaseParser::parse_value_with_unit(prop.value);
                    properties_[prop.name] = Property(value);
                }
            }
        }
    }
    
    // Set up common property aliases
    if (has_property("E") && !has_property("youngs_modulus")) {
        auto* prop = get_property("E");
        if (prop && prop->is_scalar()) {
            // Get scalar value and create new property (cannot copy unique_ptr)
            MaterialEvaluator eval;
            properties_["youngs_modulus"] = Property(prop->scalar_value(eval));
        }
    }
    if (has_property("nu") && !has_property("poisson_ratio")) {
        auto* prop = get_property("nu");
        if (prop && prop->is_scalar()) {
            MaterialEvaluator eval;
            properties_["poisson_ratio"] = Property(prop->scalar_value(eval));
        }
    }
}

Scalar Material::get_scalar(const std::string& name,
                            const MaterialEvaluator& evaluator) const {
    // Handle linearized resistivity specially
    if ((name == "electricconductivity" || name == "resistivity") && 
        linearized_resistivity_) {
        if (name == "resistivity") {
            return linearized_resistivity_->value(evaluator);
        } else {
            Scalar T = evaluator.temperature();
            return linearized_resistivity_->conductivity_at_temperature(T);
        }
    }
    
    auto* prop = get_property(name);
    if (prop) {
        return prop->scalar_value(evaluator);
    }
    
    MPFEM_WARN("Property not found: " << name << " in material " << tag_);
    return 0.0;
}

Tensor<2, 3> Material::get_conductivity(const MaterialEvaluator& evaluator) const {
    // Check for anisotropic conductivity
    auto* prop = get_property("electricconductivity");
    if (prop && prop->is_tensor()) {
        return prop->tensor_value();
    }
    
    // Use scalar conductivity (from linearized resistivity or constant)
    Scalar sigma = get_scalar("electricconductivity", evaluator);
    Tensor<2, 3> result = Tensor<2, 3>::Zero();
    result(0, 0) = sigma;
    result(1, 1) = sigma;
    result(2, 2) = sigma;
    return result;
}

Tensor<2, 3> Material::get_thermal_conductivity(const MaterialEvaluator& evaluator) const {
    auto* prop = get_property("thermalconductivity");
    if (prop && prop->is_tensor()) {
        return prop->tensor_value();
    }
    
    Scalar k = get_scalar("thermalconductivity", evaluator);
    Tensor<2, 3> result = Tensor<2, 3>::Zero();
    result(0, 0) = k;
    result(1, 1) = k;
    result(2, 2) = k;
    return result;
}

Scalar Material::get_youngs_modulus() const {
    auto* prop = get_property("youngs_modulus");
    if (prop && prop->is_scalar()) {
        // Constant property
        return prop->scalar_value(MaterialEvaluator());
    }
    auto* prop2 = get_property("E");
    if (prop2 && prop2->is_scalar()) {
        return prop2->scalar_value(MaterialEvaluator());
    }
    return 0.0;
}

Scalar Material::get_poissons_ratio() const {
    auto* prop = get_property("poisson_ratio");
    if (prop && prop->is_scalar()) {
        return prop->scalar_value(MaterialEvaluator());
    }
    auto* prop2 = get_property("nu");
    if (prop2 && prop2->is_scalar()) {
        return prop2->scalar_value(MaterialEvaluator());
    }
    return 0.0;
}

Tensor<2, 3> Material::get_thermal_expansion() const {
    auto* prop = get_property("thermalexpansioncoefficient");
    if (prop && prop->is_tensor()) {
        return prop->tensor_value();
    }
    
    Scalar alpha = 0.0;
    if (prop && prop->is_scalar()) {
        alpha = prop->scalar_value(MaterialEvaluator());
    }
    
    Tensor<2, 3> result = Tensor<2, 3>::Zero();
    result(0, 0) = alpha;
    result(1, 1) = alpha;
    result(2, 2) = alpha;
    return result;
}

Scalar Material::get_density() const {
    return get_scalar("density", MaterialEvaluator());
}

Scalar Material::get_heat_capacity() const {
    return get_scalar("heatcapacity", MaterialEvaluator());
}

void MaterialDB::build(const MaterialDatabase& config_db) {
    for (const auto& [tag, config] : config_db.materials) {
        materials_[tag] = Material(config);
    }
    MPFEM_INFO("Built material database with " << materials_.size() << " materials");
}

}  // namespace mpfem

/**
 * @file case_parser.cpp
 * @brief Parser implementation for case configuration files
 */

#include "case_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"

#include <tinyxml2.h>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <regex>

namespace mpfem {

CaseParser::CaseParser()
    : doc_(std::make_unique<tinyxml2::XMLDocument>()) {
}

CaseParser::~CaseParser() = default;

CaseConfig CaseParser::parse_case(const std::string& filename,
                                   const std::string& base_dir) {
    CaseConfig config;
    success_ = false;
    
    tinyxml2::XMLError err = doc_->LoadFile(filename.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        error_ = "Failed to load case file: " + filename;
        MPFEM_ERROR(error_);
        return config;
    }
    
    tinyxml2::XMLElement* root = doc_->RootElement();
    if (!root) {
        error_ = "Empty XML document";
        return config;
    }
    
    // Parse case name
    const char* name = root->Attribute("name");
    if (name) {
        config.name = name;
    }
    
    // Parse study type
    tinyxml2::XMLElement* study = root->FirstChildElement("study");
    if (study) {
        const char* type = study->Attribute("type");
        if (type) {
            config.study_type = type;
        }
    }
    
    // Parse paths
    tinyxml2::XMLElement* paths = root->FirstChildElement("paths");
    if (paths) {
        parse_paths(paths, config, base_dir);
    }
    
    // Parse variables
    tinyxml2::XMLElement* vars = root->FirstChildElement("variables");
    if (vars) {
        parse_variables(vars, config);
    }
    
    // Parse material assignments
    tinyxml2::XMLElement* mats = root->FirstChildElement("materials");
    if (mats) {
        parse_material_assignments(mats, config);
    }
    
    // Parse physics
    for (tinyxml2::XMLElement* physics = root->FirstChildElement("physics");
         physics != nullptr;
         physics = physics->NextSiblingElement("physics")) {
        parse_physics(physics, config);
    }
    
    // Parse coupled physics
    for (tinyxml2::XMLElement* coupled = root->FirstChildElement("coupledPhysics");
         coupled != nullptr;
         coupled = coupled->NextSiblingElement("coupledPhysics")) {
        parse_coupled_physics(coupled, config);
    }
    
    // Parse coupling configuration
    tinyxml2::XMLElement* coupling = root->FirstChildElement("coupling");
    if (coupling) {
        parse_coupling(coupling, config);
    }
    
    success_ = true;
    MPFEM_INFO("Parsed case: " << config.name 
               << " (" << config.physics.size() << " physics fields)");
    
    return config;
}

MaterialDatabase CaseParser::parse_materials(const std::string& filename) {
    MaterialDatabase db;
    success_ = false;
    
    // Create a new document for materials
    auto mat_doc = std::make_unique<tinyxml2::XMLDocument>();
    tinyxml2::XMLError err = mat_doc->LoadFile(filename.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        error_ = "Failed to load material file: " + filename;
        MPFEM_ERROR(error_);
        return db;
    }
    
    tinyxml2::XMLElement* root = mat_doc->RootElement();
    if (!root) {
        error_ = "Empty material XML document";
        return db;
    }
    
    // Navigate to model/material
    tinyxml2::XMLElement* model = root->FirstChildElement("model");
    if (!model) {
        error_ = "No model element in material file";
        return db;
    }
    
    for (tinyxml2::XMLElement* mat = model->FirstChildElement("material");
         mat != nullptr;
         mat = mat->NextSiblingElement("material")) {
        parse_material(mat, db);
    }
    
    success_ = true;
    MPFEM_INFO("Parsed " << db.materials.size() << " materials");
    
    return db;
}

void CaseParser::parse_paths(tinyxml2::XMLElement* elem, CaseConfig& config,
                              const std::string& base_dir) {
    const char* mesh = elem->Attribute("mesh");
    if (mesh) {
        if (!base_dir.empty() && mesh[0] != '/' && mesh[1] != ':') {
            config.paths.mesh = base_dir + "/" + mesh;
        } else {
            config.paths.mesh = mesh;
        }
    }
    
    const char* materials = elem->Attribute("materials");
    if (materials) {
        if (!base_dir.empty() && materials[0] != '/' && materials[1] != ':') {
            config.paths.materials = base_dir + "/" + materials;
        } else {
            config.paths.materials = materials;
        }
    }
    
    const char* comsol_result = elem->Attribute("comsol_result");
    if (comsol_result) {
        if (!base_dir.empty() && comsol_result[0] != '/' && comsol_result[1] != ':') {
            config.paths.comsol_result = base_dir + "/" + comsol_result;
        } else {
            config.paths.comsol_result = comsol_result;
        }
    }
}

void CaseParser::parse_variables(tinyxml2::XMLElement* elem, CaseConfig& config) {
    for (tinyxml2::XMLElement* var = elem->FirstChildElement("var");
         var != nullptr;
         var = var->NextSiblingElement("var")) {
        
        const char* name = var->Attribute("name");
        const char* value = var->Attribute("value");
        const char* si = var->Attribute("si");
        
        if (name && value) {
            Variable v;
            v.name = name;
            v.value = value;
            
            if (si) {
                v.si_value = std::stod(si);
            } else {
                v.si_value = parse_value_with_unit(value);
            }
            
            config.variables[name] = v;
        }
    }
}

void CaseParser::parse_material_assignments(tinyxml2::XMLElement* elem,
                                             CaseConfig& config) {
    for (tinyxml2::XMLElement* assign = elem->FirstChildElement("assign");
         assign != nullptr;
         assign = assign->NextSiblingElement("assign")) {
        
        const char* domains = assign->Attribute("domains");
        const char* material = assign->Attribute("material");
        
        if (domains && material) {
            MaterialAssignment ma;
            ma.domains = parse_id_list(domains);
            ma.material = material;
            config.material_assignments.push_back(ma);
        }
    }
}

void CaseParser::parse_physics(tinyxml2::XMLElement* elem, CaseConfig& config) {
    PhysicsConfig pc;
    
    const char* kind = elem->Attribute("kind");
    if (kind) {
        pc.kind = kind;
    }
    
    int order = 1;
    elem->QueryIntAttribute("order", &order);
    pc.order = order;
    
    // Parse solver
    tinyxml2::XMLElement* solver = elem->FirstChildElement("solver");
    if (solver) {
        parse_solver(solver, pc.solver);
    }
    
    // Parse boundary conditions
    for (tinyxml2::XMLElement* bc = elem->FirstChildElement("boundary");
         bc != nullptr;
         bc = bc->NextSiblingElement("boundary")) {
        parse_boundary_condition(bc, pc);
    }
    
    // Parse sources
    for (tinyxml2::XMLElement* src = elem->FirstChildElement("source");
         src != nullptr;
         src = src->NextSiblingElement("source")) {
        parse_source(src, pc);
    }
    
    config.physics.push_back(pc);
}

void CaseParser::parse_boundary_condition(tinyxml2::XMLElement* elem,
                                           PhysicsConfig& physics) {
    BoundaryConditionConfig bc;
    
    const char* kind = elem->Attribute("kind");
    const char* ids = elem->Attribute("ids");
    
    if (kind) bc.kind = kind;
    if (ids) bc.ids = parse_id_list(ids);
    
    // Parse parameters
    for (tinyxml2::XMLElement* param = elem->FirstChildElement("param");
         param != nullptr;
         param = param->NextSiblingElement("param")) {
        
        const char* name = param->Attribute("name");
        const char* value = param->Attribute("value");
        
        if (name && value) {
            bc.params[name] = value;
        }
    }
    
    physics.boundaries.push_back(bc);
}

void CaseParser::parse_source(tinyxml2::XMLElement* elem, PhysicsConfig& physics) {
    SourceConfig src;
    
    const char* kind = elem->Attribute("kind");
    const char* domains = elem->Attribute("domains");
    const char* value = elem->Attribute("value");
    
    if (kind) src.kind = kind;
    if (domains) src.domains = parse_id_list(domains);
    if (value) src.value = value;
    
    physics.sources.push_back(src);
}

void CaseParser::parse_solver(tinyxml2::XMLElement* elem, SolverConfig& solver) {
    const char* type = elem->Attribute("type");
    if (type) solver.type = type;
    
    elem->QueryIntAttribute("max_iter", &solver.max_iter);
    
    double tol = 1e-10;
    if (elem->QueryDoubleAttribute("tolerance", &tol) == tinyxml2::XML_SUCCESS) {
        solver.tolerance = tol;
    }
    
    elem->QueryIntAttribute("print_level", &solver.print_level);
}

void CaseParser::parse_coupled_physics(tinyxml2::XMLElement* elem,
                                        CaseConfig& config) {
    CoupledPhysicsConfig cp;
    
    const char* name = elem->Attribute("name");
    const char* kind = elem->Attribute("kind");
    const char* physics = elem->Attribute("physics");
    const char* domains = elem->Attribute("domains");
    
    if (name) cp.name = name;
    if (kind) cp.kind = kind;
    if (physics) {
        // Split by comma
        std::string phys_str = physics;
        std::stringstream ss(phys_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            cp.physics.push_back(token);
        }
    }
    if (domains) cp.domains = parse_id_list(domains);
    
    config.coupled_physics.push_back(cp);
}

void CaseParser::parse_coupling(tinyxml2::XMLElement* elem, CaseConfig& config) {
    const char* method = elem->Attribute("method");
    if (method) config.coupling.method = method;
    
    elem->QueryIntAttribute("max_iter", &config.coupling.max_iter);
    
    double tol = 1e-8;
    if (elem->QueryDoubleAttribute("tolerance", &tol) == tinyxml2::XML_SUCCESS) {
        config.coupling.tolerance = tol;
    }
}

void CaseParser::parse_material(tinyxml2::XMLElement* elem, MaterialDatabase& db) {
    MaterialConfig mat;
    
    const char* tag = elem->Attribute("tag");
    if (tag) mat.tag = tag;
    
    // Parse label
    for (tinyxml2::XMLElement* label = elem->FirstChildElement("label");
         label != nullptr;
         label = label->NextSiblingElement("label")) {
        const char* lbl = label->Attribute("label");
        if (lbl) mat.label = lbl;
    }
    
    // Parse property groups
    for (tinyxml2::XMLElement* group = elem->FirstChildElement("propertyGroup");
         group != nullptr;
         group = group->NextSiblingElement("propertyGroup")) {
        parse_property_group(group, mat);
    }
    
    // Parse global set elements
    for (tinyxml2::XMLElement* set = elem->FirstChildElement("set");
         set != nullptr;
         set = set->NextSiblingElement("set")) {
        const char* name = set->Attribute("name");
        const char* value = set->Attribute("value");
        if (name && value) {
            MaterialProperty prop;
            prop.name = name;
            prop.value = value;
            prop.is_tensor = (value[0] == '{');
            mat.properties.push_back(prop);
        }
    }
    
    // Parse inputs
    for (tinyxml2::XMLElement* input = elem->FirstChildElement("addInput");
         input != nullptr;
         input = input->NextSiblingElement("addInput")) {
        const char* qty = input->Attribute("quantity");
        if (qty) {
            mat.inputs.push_back(qty);
        }
    }
    
    db.add(mat);
}

void CaseParser::parse_property_group(tinyxml2::XMLElement* elem,
                                       MaterialConfig& mat) {
    const char* tag = elem->Attribute("tag");
    if (!tag) return;
    
    std::vector<MaterialProperty> props;
    
    for (tinyxml2::XMLElement* set = elem->FirstChildElement("set");
         set != nullptr;
         set = set->NextSiblingElement("set")) {
        const char* name = set->Attribute("name");
        const char* value = set->Attribute("value");
        if (name && value) {
            MaterialProperty prop;
            prop.name = name;
            prop.value = value;
            prop.is_tensor = (value[0] == '{');
            props.push_back(prop);
        }
    }
    
    mat.property_groups[tag] = props;
}

Scalar CaseParser::evaluate_expression(const std::string& expr,
                                        const CaseConfig& config) {
    // Simple expression evaluation with variable substitution
    // Supports: variable names, numbers, basic arithmetic (+, -, *, /)
    
    // Check if it's a simple variable name
    auto it = config.variables.find(expr);
    if (it != config.variables.end()) {
        return it->second.si_value;
    }
    
    // Check for value with unit
    if (expr.find('[') != std::string::npos) {
        return parse_value_with_unit(expr);
    }
    
    // TODO: Implement full expression parser for complex expressions
    // For now, try to parse as a number
    try {
        return std::stod(expr);
    } catch (...) {
        MPFEM_WARN("Could not evaluate expression: " << expr);
        return 0.0;
    }
}

Scalar CaseParser::parse_value_with_unit(const std::string& value_str) {
    // Parse value with unit like "9[cm]", "5[W/m^2/K]", "0.02"
    
    // Find unit brackets
    size_t bracket_pos = value_str.find('[');
    if (bracket_pos == std::string::npos) {
        // No unit, just a number
        try {
            return std::stod(value_str);
        } catch (...) {
            return 0.0;
        }
    }
    
    // Extract value and unit
    Scalar value = std::stod(value_str.substr(0, bracket_pos));
    std::string unit = value_str.substr(bracket_pos + 1);
    
    // Remove closing bracket
    size_t end = unit.find(']');
    if (end != std::string::npos) {
        unit = unit.substr(0, end);
    }
    
    // Unit conversion factors
    static const std::unordered_map<std::string, Scalar> unit_factors = {
        // Length
        {"m", 1.0},
        {"cm", 0.01},
        {"mm", 0.001},
        // Time
        {"s", 1.0},
        // Mass
        {"kg", 1.0},
        // Temperature
        {"K", 1.0},
        // Voltage
        {"V", 1.0},
        {"mV", 0.001},
        // Pressure/Stress
        {"Pa", 1.0},
        {"kPa", 1000.0},
        {"MPa", 1e6},
        {"GPa", 1e9},
        // Thermal
        {"W/m^2/K", 1.0},
        {"W/(m*K)", 1.0},
        {"W/(m*K)", 1.0},
        {"J/(kg*K)", 1.0},
        // Electrical
        {"S/m", 1.0},
        {"ohm*m", 1.0},
        // Dimensionless
        {"1/K", 1.0},
    };
    
    auto it = unit_factors.find(unit);
    if (it != unit_factors.end()) {
        return value * it->second;
    }
    
    // Unknown unit, return value as-is with warning
    MPFEM_WARN("Unknown unit: " << unit << ", treating as dimensionless");
    return value;
}

std::vector<Index> CaseParser::parse_id_list(const std::string& id_str) {
    std::vector<Index> ids;
    
    // Split by comma
    std::stringstream ss(id_str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        // Check for range (e.g., "1-7")
        size_t dash = token.find('-');
        if (dash != std::string::npos && dash > 0) {
            Index start = std::stoll(token.substr(0, dash));
            Index end = std::stoll(token.substr(dash + 1));
            for (Index i = start; i <= end; ++i) {
                ids.push_back(i);
            }
        } else {
            // Single value
            ids.push_back(std::stoll(token));
        }
    }
    
    return ids;
}

std::vector<Scalar> CaseParser::parse_tensor(const std::string& tensor_str) {
    std::vector<Scalar> values;
    
    // Parse format: "{'v1','v2','v3',...}"
    if (tensor_str.empty() || tensor_str[0] != '{') {
        return values;
    }
    
    // Remove braces
    std::string content = tensor_str.substr(1, tensor_str.size() - 2);
    
    // Split by comma
    std::stringstream ss(content);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        // Remove quotes and whitespace
        token.erase(std::remove(token.begin(), token.end(), '\''), token.end());
        token.erase(std::remove(token.begin(), token.end(), '"'), token.end());
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        // Check for unit
        if (token.find('[') != std::string::npos) {
            values.push_back(parse_value_with_unit(token));
        } else {
            values.push_back(std::stod(token));
        }
    }
    
    return values;
}

}  // namespace mpfem

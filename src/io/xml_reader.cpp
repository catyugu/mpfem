/**
 * @file xml_reader.cpp
 * @brief Implementation of XML configuration reader
 */

#include "xml_reader.hpp"
#include "core/logger.hpp"

#include <tinyxml2.h>
#include <sstream>
#include <algorithm>

namespace mpfem {

XmlReader::XmlReader() 
    : doc_(std::make_unique<tinyxml2::XMLDocument>()) {
}

XmlReader::~XmlReader() = default;

CaseConfig XmlReader::read(const std::string& filename) {
    CaseConfig config;
    
    tinyxml2::XMLError err = doc_->LoadFile(filename.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        error_ = "Failed to load XML file: " + filename;
        MPFEM_ERROR(error_);
        return config;
    }
    
    tinyxml2::XMLElement* root = doc_->RootElement();
    if (!root) {
        error_ = "Empty XML document";
        MPFEM_ERROR(error_);
        return config;
    }
    
    // Get case name
    const char* name_attr = root->Attribute("name");
    if (name_attr) {
        config.name = name_attr;
    }
    
    // Parse paths
    tinyxml2::XMLElement* paths = root->FirstChildElement("paths");
    if (paths) {
        parse_paths(paths, config);
    }
    
    // Parse study type
    tinyxml2::XMLElement* study = root->FirstChildElement("study");
    if (study) {
        const char* type = study->Attribute("type");
        if (type) {
            config.study_type = type;
        }
    }
    
    // Parse variables
    tinyxml2::XMLElement* variables = root->FirstChildElement("variables");
    if (variables) {
        parse_variables(variables, config);
    }
    
    // Parse materials
    tinyxml2::XMLElement* materials = root->FirstChildElement("materials");
    if (materials) {
        parse_materials(materials, config);
    }
    
    // Parse physics fields
    for (tinyxml2::XMLElement* physics = root->FirstChildElement("physics");
         physics != nullptr;
         physics = physics->NextSiblingElement("physics")) {
        parse_physics(physics, config);
    }
    
    // Parse coupled physics
    for (tinyxml2::XMLElement* coupled = root->FirstChildElement("coupledPhysics");
         coupled != nullptr;
         coupled = coupled->NextSiblingElement("coupledPhysics")) {
        parse_couplings(coupled, config);
    }
    
    // Parse nonlinear solver config
    tinyxml2::XMLElement* coupling = root->FirstChildElement("coupling");
    if (coupling) {
        const char* method = coupling->Attribute("method");
        if (method) {
            config.nonlinear.method = method;
        }
        
        int max_iter = 0;
        if (coupling->QueryIntAttribute("max_iter", &max_iter) == tinyxml2::XML_SUCCESS) {
            config.nonlinear.max_iter = max_iter;
        }
        
        double tolerance = 0;
        if (coupling->QueryDoubleAttribute("tolerance", &tolerance) == tinyxml2::XML_SUCCESS) {
            config.nonlinear.tolerance = static_cast<Scalar>(tolerance);
        }
    }
    
    MPFEM_INFO("Loaded case configuration: " << config.name);
    MPFEM_INFO("  - " << config.physics.size() << " physics fields");
    MPFEM_INFO("  - " << config.couplings.size() << " couplings");
    MPFEM_INFO("  - " << config.material_assignments.size() << " material assignments");
    
    return config;
}

void XmlReader::parse_paths(tinyxml2::XMLElement* elem, CaseConfig& config) {
    const char* mesh = elem->Attribute("mesh");
    if (mesh) {
        config.mesh_path = mesh;
    }
    
    const char* materials = elem->Attribute("materials");
    if (materials) {
        config.material_path = materials;
    }
    
    const char* result = elem->Attribute("comsol_result");
    if (result) {
        config.result_path = result;
    }
}

void XmlReader::parse_variables(tinyxml2::XMLElement* elem, CaseConfig& config) {
    for (tinyxml2::XMLElement* var = elem->FirstChildElement("var");
         var != nullptr;
         var = var->NextSiblingElement("var")) {
        
        const char* name = var->Attribute("name");
        const char* value = var->Attribute("value");
        
        if (name && value) {
            // Try to use pre-computed SI value if available
            const char* si = var->Attribute("si");
            if (si) {
                try {
                    Scalar si_value = std::stod(si);
                    config.variables.set_variable(name, si_value);
                } catch (...) {
                    // Fall back to parsing
                    config.variables.parse_variable(name, value);
                }
            } else {
                config.variables.parse_variable(name, value);
            }
        }
    }
    
    MPFEM_INFO("Parsed " << config.variables.variable_names().size() << " variables");
}

void XmlReader::parse_materials(tinyxml2::XMLElement* elem, CaseConfig& config) {
    for (tinyxml2::XMLElement* assign = elem->FirstChildElement("assign");
         assign != nullptr;
         assign = assign->NextSiblingElement("assign")) {
        
        const char* domains = assign->Attribute("domains");
        const char* material = assign->Attribute("material");
        
        if (domains && material) {
            MaterialAssignment ma;
            ma.domains = parse_domain_list(domains);
            ma.material_tag = material;
            config.material_assignments.push_back(std::move(ma));
        }
    }
}

void XmlReader::parse_physics(tinyxml2::XMLElement* elem, CaseConfig& config) {
    PhysicsConfig pc;
    
    const char* kind = elem->Attribute("kind");
    if (kind) {
        pc.kind = kind;
    }
    
    int order = 1;
    if (elem->QueryIntAttribute("order", &order) == tinyxml2::XML_SUCCESS) {
        pc.order = order;
    }
    
    // Parse solver config
    tinyxml2::XMLElement* solver = elem->FirstChildElement("solver");
    if (solver) {
        pc.solver = parse_solver(solver);
    }
    
    // Parse boundary conditions
    for (tinyxml2::XMLElement* bc = elem->FirstChildElement("boundary");
         bc != nullptr;
         bc = bc->NextSiblingElement("boundary")) {
        pc.boundaries.push_back(parse_boundary(bc));
    }
    
    // Parse source terms
    for (tinyxml2::XMLElement* src = elem->FirstChildElement("source");
         src != nullptr;
         src = src->NextSiblingElement("source")) {
        pc.sources.push_back(parse_source(src));
    }
    
    // Store with index for quick lookup
    config.physics_index[pc.kind] = config.physics.size();
    config.physics.push_back(std::move(pc));
}

void XmlReader::parse_couplings(tinyxml2::XMLElement* elem, CaseConfig& config) {
    CouplingConfig cc;
    
    const char* name = elem->Attribute("name");
    if (name) {
        cc.name = name;
    }
    
    const char* kind = elem->Attribute("kind");
    if (kind) {
        cc.kind = kind;
    }
    
    const char* physics = elem->Attribute("physics");
    if (physics) {
        // Split by comma
        std::istringstream ss(physics);
        std::string p;
        while (std::getline(ss, p, ',')) {
            // Trim whitespace
            p.erase(0, p.find_first_not_of(" \t"));
            p.erase(p.find_last_not_of(" \t") + 1);
            if (!p.empty()) {
                cc.physics.push_back(p);
            }
        }
    }
    
    const char* domains = elem->Attribute("domains");
    if (domains) {
        cc.domains = parse_domain_list(domains);
    }
    
    config.couplings.push_back(std::move(cc));
}

BoundaryConditionConfig XmlReader::parse_boundary(tinyxml2::XMLElement* elem) {
    BoundaryConditionConfig bc;
    
    const char* kind = elem->Attribute("kind");
    if (kind) {
        bc.kind = kind;
    }
    
    const char* ids = elem->Attribute("ids");
    if (ids) {
        bc.ids = parse_id_list(ids);
    }
    
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
    
    return bc;
}

SourceConfig XmlReader::parse_source(tinyxml2::XMLElement* elem) {
    SourceConfig sc;
    
    const char* kind = elem->Attribute("kind");
    if (kind) {
        sc.kind = kind;
    }
    
    const char* domains = elem->Attribute("domains");
    if (domains) {
        sc.domains = parse_domain_list(domains);
    }
    
    const char* value = elem->Attribute("value");
    if (value) {
        sc.value = value;
    }
    
    return sc;
}

SolverConfig XmlReader::parse_solver(tinyxml2::XMLElement* elem) {
    SolverConfig sc;
    
    const char* type = elem->Attribute("type");
    if (type) {
        sc.type = type;
    }
    
    int max_iter = 0;
    if (elem->QueryIntAttribute("max_iter", &max_iter) == tinyxml2::XML_SUCCESS) {
        sc.max_iter = max_iter;
    }
    
    double tolerance = 0;
    if (elem->QueryDoubleAttribute("tolerance", &tolerance) == tinyxml2::XML_SUCCESS) {
        sc.tolerance = static_cast<Scalar>(tolerance);
    }
    
    int print_level = 0;
    if (elem->QueryIntAttribute("print_level", &print_level) == tinyxml2::XML_SUCCESS) {
        sc.print_level = print_level;
    }
    
    return sc;
}

std::vector<Index> XmlReader::parse_id_list(const std::string& id_str) const {
    std::vector<Index> ids;
    
    // Parse format: "1-7,9-14,16-42" or "1,2,3"
    std::istringstream ss(id_str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        if (token.empty()) continue;
        
        // Check for range
        size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            // Range: start-end
            Index start = std::stoll(token.substr(0, dash_pos));
            Index end = std::stoll(token.substr(dash_pos + 1));
            
            for (Index i = start; i <= end; ++i) {
                ids.push_back(i);
            }
        } else {
            // Single ID
            ids.push_back(std::stoll(token));
        }
    }
    
    return ids;
}

std::vector<Index> XmlReader::parse_domain_list(const std::string& domain_str) const {
    return parse_id_list(domain_str);
}

} // namespace mpfem

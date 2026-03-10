/**
 * @file case_config.hpp
 * @brief Configuration data structures for simulation cases
 */

#ifndef MPFEM_CONFIG_CASE_CONFIG_HPP
#define MPFEM_CONFIG_CASE_CONFIG_HPP

#include "core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace mpfem {

/**
 * @brief Variable definition
 */
struct Variable {
    std::string name;
    std::string value;      ///< Value with unit (e.g., "9[cm]")
    Scalar si_value;        ///< Value in SI units
};

/**
 * @brief Boundary condition configuration
 */
struct BoundaryConditionConfig {
    std::string kind;       ///< BC type (voltage, convection, fixed_constraint, etc.)
    std::vector<Index> ids; ///< Boundary IDs
    
    /// Parameters (name -> value expression)
    std::unordered_map<std::string, std::string> params;
};

/**
 * @brief Source term configuration
 */
struct SourceConfig {
    std::string kind;           ///< Source type
    std::vector<Index> domains; ///< Domain IDs
    std::string value;          ///< Value expression or source name
};

/**
 * @brief Solver configuration
 */
struct SolverConfig {
    std::string type = "direct";    ///< Solver type: "direct", "cg", "bicgstab"
    int max_iter = 1000;
    Scalar tolerance = 1e-10;
    int print_level = 0;
};

/**
 * @brief Physics field configuration
 */
struct PhysicsConfig {
    std::string kind;       ///< Physics type: electrostatics, heat_transfer, solid_mechanics
    int order = 1;          ///< Element order
    SolverConfig solver;
    
    std::vector<BoundaryConditionConfig> boundaries;
    std::vector<SourceConfig> sources;
};

/**
 * @brief Coupled physics configuration
 */
struct CoupledPhysicsConfig {
    std::string name;       ///< Coupling name
    std::string kind;       ///< Coupling type: joule_heating, thermal_expansion
    std::vector<std::string> physics; ///< Names of coupled physics
    std::vector<Index> domains;
};

/**
 * @brief Coupling solver configuration
 */
struct CouplingConfig {
    std::string method = "picard";  ///< Coupling method
    int max_iter = 20;
    Scalar tolerance = 1e-8;
};

/**
 * @brief Material property value
 */
struct MaterialProperty {
    std::string name;
    std::string value;      ///< Value expression or tensor string
    bool is_tensor = false; ///< True if value is a tensor
};

/**
 * @brief Material definition
 */
struct MaterialConfig {
    std::string tag;        ///< Material tag (e.g., "mat1")
    std::string label;      ///< Material name (e.g., "Copper")
    
    /// Property groups (group_tag -> properties)
    std::unordered_map<std::string, std::vector<MaterialProperty>> property_groups;
    
    /// Global properties
    std::vector<MaterialProperty> properties;
    
    /// Field dependencies (e.g., "temperature")
    std::vector<std::string> inputs;
};

/**
 * @brief Material assignment
 */
struct MaterialAssignment {
    std::vector<Index> domains; ///< Domain IDs
    std::string material;       ///< Material tag
};

/**
 * @brief Path configuration
 */
struct PathConfig {
    std::string mesh;
    std::string materials;
    std::string comsol_result;
};

/**
 * @brief Complete case configuration
 */
struct CaseConfig {
    std::string name;
    std::string study_type = "steady";  ///< steady, transient, eigenvalue
    
    PathConfig paths;
    
    /// Variables (name -> Variable)
    std::unordered_map<std::string, Variable> variables;
    
    /// Material assignments
    std::vector<MaterialAssignment> material_assignments;
    
    /// Physics fields
    std::vector<PhysicsConfig> physics;
    
    /// Coupled physics
    std::vector<CoupledPhysicsConfig> coupled_physics;
    
    /// Coupling solver configuration
    CouplingConfig coupling;
    
    // Helper methods
    
    /**
     * @brief Get variable value by name
     * @return Variable value, or 0 if not found
     */
    Scalar get_variable(const std::string& name) const {
        auto it = variables.find(name);
        if (it != variables.end()) {
            return it->second.si_value;
        }
        return 0.0;
    }
    
    /**
     * @brief Check if variable exists
     */
    bool has_variable(const std::string& name) const {
        return variables.find(name) != variables.end();
    }
    
    /**
     * @brief Get physics config by kind
     */
    const PhysicsConfig* get_physics(const std::string& kind) const {
        for (const auto& p : physics) {
            if (p.kind == kind) return &p;
        }
        return nullptr;
    }
};

/**
 * @brief Material database
 */
struct MaterialDatabase {
    /// Materials by tag
    std::unordered_map<std::string, MaterialConfig> materials;
    
    /**
     * @brief Get material by tag
     */
    const MaterialConfig* get(const std::string& tag) const {
        auto it = materials.find(tag);
        if (it != materials.end()) return &it->second;
        return nullptr;
    }
    
    /**
     * @brief Add material
     */
    void add(const MaterialConfig& mat) {
        materials[mat.tag] = mat;
    }
};

}  // namespace mpfem

#endif  // MPFEM_CONFIG_CASE_CONFIG_HPP

#ifndef MPFEM_CASE_DEFINITION_HPP
#define MPFEM_CASE_DEFINITION_HPP

#include "model/field_kind.hpp"
#include "solver/solver_config.hpp"
#include "core/exception.hpp"
#include <string>
#include <vector>
#include <set>
#include <map>

namespace mpfem {

/**
 * @brief Scalar variable definition from case XML.
 */
struct VariableEntry {
    std::string name;
    std::string valueText;
    double siValue = 0.0;
};

/**
 * @brief Domain to material tag mapping rule.
 */
struct MaterialAssignment {
    std::set<int> domainIds;
    std::string materialTag;
};

/**
 * @brief Boundary condition definition for one physics block.
 * 
 * Parameters are stored as key-value pairs. Different boundary condition types
 * require different parameters.
 */
struct BoundaryCondition {
    std::string kind;
    std::set<int> ids;
    std::map<std::string, std::string> params;  // param name -> value expression
};

/**
 * @brief Volumetric or surface source definition.
 */
struct SourceDefinition {
    std::string kind;
    std::set<int> domainIds;
    std::string valueText;
};

/**
 * @brief Physics block extracted from case XML.
 */
struct PhysicsDefinition {
    std::string kind;
    int order = 1;  // Polynomial order for this physics field
    SolverConfig solver;  // Use SolverConfig from solver/solver_config.hpp
    std::vector<BoundaryCondition> boundaries;
    std::vector<SourceDefinition> sources;
};

/**
 * @brief Coupling iteration configuration.
 */
struct CouplingConfig {
    int maxIterations = 15;
    double tolerance = 1e-6;
};

/**
 * @brief Cross-physics coupling definition extracted from case XML.
 */
struct CoupledPhysicsDefinition {
    std::string name;
    std::string kind;
    std::vector<std::string> physicsKinds;
    std::set<int> domainIds;
};

/**
 * @brief Fully parsed case configuration.
 */
struct CaseDefinition {
    std::string caseName;
    std::string studyType;
    std::string meshPath;
    std::string materialsPath;
    std::string comsolResultPath;
    std::vector<VariableEntry> variables;
    std::vector<MaterialAssignment> materialAssignments;
    std::vector<PhysicsDefinition> physicsDefinitions;
    std::vector<CoupledPhysicsDefinition> coupledPhysicsDefinitions;
    CouplingConfig couplingConfig;

    /**
     * @brief Get variable value by name.
     * @throws Exception if variable not found.
     */
    double getVariable(const std::string& name) const {
        for (const auto& v : variables) {
            if (v.name == name) {
                return v.siValue;
            }
        }
        MPFEM_THROW(Exception, 
            "CaseDefinition::getVariable: variable '" + name + "' not found");
    }
    
    /**
     * @brief Check if variable exists.
     */
    bool hasVariable(const std::string& name) const {
        for (const auto& v : variables) {
            if (v.name == name) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Get variable value by name, or default if not found.
     */
    double getVariableOrDefault(const std::string& name, double defaultValue) const {
        for (const auto& v : variables) {
            if (v.name == name) {
                return v.siValue;
            }
        }
        return defaultValue;
    }

    /**
     * @brief Build variable map for value resolution.
     */
    std::map<std::string, double> getVariableMap() const {
        std::map<std::string, double> result;
        for (const auto& v : variables) {
            result[v.name] = v.siValue;
        }
        return result;
    }
};

/**
 * @brief Boundary condition parameters (resolved to numeric values).
 */
struct BoundaryParams {
    std::string kind;
    std::map<std::string, double> values;
};

/// Boundary conditions organized by boundary id for each physics field
using BoundaryConditions = std::map<int, BoundaryParams>;

/**
 * @brief Field source definition (resolved).
 */
struct FieldSource {
    FieldKind field = FieldKind::Unknown;
    std::set<int> domainIds;
    bool coupled = false;
    CouplingKind couplingKind = CouplingKind::Unknown;
    double value = 0.0;
};

/**
 * @brief Field configuration (resolved).
 */
struct FieldConfig {
    int order = 1;
    SolverConfig solver;  // Use SolverConfig from solver/solver_config.hpp
};

/**
 * @brief Resolved physics problem model.
 */
struct PhysicsProblemModel {
    std::string caseName;
    std::string studyType;
    std::string meshPath;
    std::string comsolResultPath;
    std::map<std::string, double> variables;
    std::map<int, std::string> domainMaterialTag;
    std::map<FieldKind, BoundaryConditions> boundaries;  // field -> id -> params
    std::vector<FieldSource> sources;
    std::vector<CouplingKind> couplings;
    std::map<FieldKind, FieldConfig> fieldConfigs;
    CouplingConfig couplingConfig;
};

}  // namespace mpfem

#endif  // MPFEM_CASE_DEFINITION_HPP
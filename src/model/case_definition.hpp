#ifndef MPFEM_CASE_DEFINITION_HPP
#define MPFEM_CASE_DEFINITION_HPP

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
 * @brief Time configuration for time-dependent simulations.
 */
struct TimeConfig {
    double start = 0.0;
    double end = 1.0;
    double step = 0.01;
    std::string scheme = "BDF1";  // BDF1, BDF2, CrankNicolson
};

/**
 * @brief Initial condition definition for a physics field.
 */
struct InitialCondition {
    std::string fieldKind;  // "electrostatics", "heat_transfer", "solid_mechanics"
    double value = 0.0;     // scalar value for initial condition
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
    /**
     * Physics block keyed by physics kind (e.g., "electrostatics", "heat_transfer").
     */
    struct Physics {
        std::string kind;
        int order = 1;  // Polynomial order for this physics field
        SolverConfig solver;
        std::vector<BoundaryCondition> boundaries;
        std::vector<SourceDefinition> sources;
        // Reference temperature for thermal expansion [K]
        double referenceTemperature = 293.15;
    };

    std::string caseName;
    std::string studyType;
    std::string meshPath;
    std::string materialsPath;
    std::string comsolResultPath;
    std::vector<VariableEntry> variables;
    std::vector<MaterialAssignment> materialAssignments;
    std::map<std::string, Physics> physics;  // keyed by physics kind
    std::vector<CoupledPhysicsDefinition> coupledPhysicsDefinitions;
    CouplingConfig couplingConfig;
    TimeConfig timeConfig;
    std::vector<InitialCondition> initialConditions;

    // O(1) variable lookup map, built explicitly after variables are populated.
    std::map<std::string, double> variableMap_;

    /**
     * @brief Build variable map for O(1) lookup from variables vector.
     * Call this after variables are populated or modified.
     */
    void buildVariableMap() {
        variableMap_.clear();
        for (const auto& v : variables) {
            variableMap_[v.name] = v.siValue;
        }
    }

    /**
     * @brief Get variable value by name.
     * @throws Exception if variable not found.
     */
    double getVariable(const std::string& name) const {
        auto it = variableMap_.find(name);
        if (it != variableMap_.end()) {
            return it->second;
        }
        MPFEM_THROW(Exception, 
            "CaseDefinition::getVariable: variable '" + name + "' not found");
    }
    
    /**
     * @brief Check if variable exists.
     */
    bool hasVariable(const std::string& name) const {
        return variableMap_.find(name) != variableMap_.end();
    }
    
};

}  // namespace mpfem

#endif  // MPFEM_CASE_DEFINITION_HPP
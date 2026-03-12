#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "electrostatics_solver.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "core/logger.hpp"
#include <memory>
#include <map>
#include <string>

namespace mpfem {

/**
 * @file physics_problem_builder.hpp
 * @brief Builder class for constructing physics problems from configuration.
 * 
 * Takes a case definition and builds all necessary solver objects.
 */

/**
 * @brief Result of building a physics problem.
 */
struct PhysicsProblemSetup {
    std::string caseName;
    std::unique_ptr<Mesh> mesh;
    MaterialDatabase materials;
    CaseDefinition caseDef;
    
    // Solvers
    std::unique_ptr<ElectrostaticsSolver> electrostatics;
    // std::unique_ptr<HeatTransferSolver> heatTransfer;  // TODO
    // std::unique_ptr<SolidMechanicsSolver> solidMechanics;  // TODO
    
    // Coupling coefficients
    std::unique_ptr<JouleHeatCoefficient> jouleHeatSource;
    
    bool hasElectrostatics() const { return electrostatics != nullptr; }
    // bool hasHeatTransfer() const { return heatTransfer != nullptr; }
    // bool hasSolidMechanics() const { return solidMechanics != nullptr; }
};

/**
 * @brief Builder for physics problems.
 * 
 * Reads case configuration and builds all solver objects.
 */
class PhysicsProblemBuilder {
public:
    /**
     * @brief Build a complete physics problem from case directory.
     * @param caseDir Path to case directory containing case.xml, material.xml, mesh
     * @return Setup containing all built objects
     */
    static PhysicsProblemSetup build(const std::string& caseDir) {
        PhysicsProblemSetup setup;
        
        // Read case definition
        std::string casePath = caseDir + "/case.xml";
        LOG_INFO << "Reading case from " << casePath;
        
        CaseXmlReader::readFromFile(casePath, setup.caseDef);
        setup.caseName = setup.caseDef.caseName;
        
        // Read mesh
        std::string meshPath = caseDir + "/" + setup.caseDef.meshPath;
        LOG_INFO << "Reading mesh from " << meshPath;
        
        setup.mesh = std::make_unique<Mesh>(MphtxtReader::read(meshPath));
        LOG_INFO << "Mesh loaded: " << setup.mesh->numVertices() << " vertices, "
                 << setup.mesh->numElements() << " elements";
        
        // Read materials
        std::string matPath = caseDir + "/" + setup.caseDef.materialsPath;
        LOG_INFO << "Reading materials from " << matPath;
        
        MaterialXmlReader::readFromFile(matPath, setup.materials);
        
        // Build solvers for each physics field
        buildSolvers(setup);
        
        return setup;
    }
    
private:
    static void buildSolvers(PhysicsProblemSetup& setup) {
        const auto& caseDef = setup.caseDef;
        const auto& mesh = *setup.mesh;
        const auto& materials = setup.materials;
        
        // Build domain material map
        std::map<int, std::string> domainMaterial;
        for (const auto& assign : caseDef.materialAssignments) {
            for (int domId : assign.domainIds) {
                domainMaterial[domId] = assign.materialTag;
            }
        }
        
        // Find maximum domain ID
        int maxDomainId = 0;
        for (const auto& [domId, _] : domainMaterial) {
            maxDomainId = std::max(maxDomainId, domId);
        }
        
        // Build each physics field
        for (const auto& physics : caseDef.physicsDefinitions) {
            if (physics.kind == "electrostatics") {
                buildElectrostatics(setup, physics, domainMaterial, maxDomainId, materials);
            }
            // else if (physics.kind == "heat_transfer") {
            //     buildHeatTransfer(setup, physics, domainMaterial, maxDomainId, materials);
            // }
            // else if (physics.kind == "solid_mechanics") {
            //     buildSolidMechanics(setup, physics, domainMaterial, maxDomainId, materials);
            // }
            else {
                LOG_WARN << "Unknown physics kind: " << physics.kind;
            }
        }
    }
    
    static void buildElectrostatics(
        PhysicsProblemSetup& setup,
        const PhysicsDefinition& physics,
        const std::map<int, std::string>& domainMaterial,
        int maxDomainId,
        const MaterialDatabase& materials)
    {
        LOG_INFO << "Building electrostatics solver, order = " << physics.order;
        
        // Create conductivity coefficient
        auto conductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
        
        for (const auto& [domId, matTag] : domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat) {
                double sigma = mat->electricConductivity;
                conductivity->setConstant(domId, sigma);
                LOG_INFO << "Domain " << domId << " (" << matTag << "): sigma = " << sigma;
            } else {
                LOG_WARN << "Material '" << matTag << "' not found for domain " << domId;
            }
        }
        
        // Create solver
        setup.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
        setup.electrostatics->setSolverType(physics.solver.type);
        setup.electrostatics->setMaxIterations(physics.solver.maxIterations);
        setup.electrostatics->setTolerance(physics.solver.relativeTolerance);
        setup.electrostatics->setPrintLevel(physics.solver.printLevel);
        
        // Initialize
        setup.electrostatics->initialize(*setup.mesh, *conductivity);
        
        // Apply boundary conditions
        for (const auto& bc : physics.boundaries) {
            if (bc.kind == "voltage") {
                // Get voltage value
                Real value = 0.0;
                auto it = bc.params.find("value");
                if (it != bc.params.end()) {
                    // Try to parse as number or variable reference
                    value = parseValue(it->second, setup.caseDef);
                }
                
                for (int id : bc.ids) {
                    setup.electrostatics->addDirichletBC(id, value);
                    LOG_INFO << "BC: voltage = " << value << " V on boundary " << id;
                }
            }
            else if (bc.kind == "electric_insulation") {
                // Natural BC (do nothing for Neumann zero)
                LOG_DEBUG << "BC: electric insulation on boundaries " 
                          << joinIds(bc.ids);
            }
            else {
                LOG_WARN << "Unknown BC kind for electrostatics: " << bc.kind;
            }
        }
        
        // Store conductivity for later use (e.g., Joule heating)
        // setup.conductivity = std::move(conductivity);
    }
    
    static Real parseValue(const std::string& str, const CaseDefinition& caseDef) {
        // Remove units if present (e.g., "0[V]" -> "0")
        std::string numStr = str;
        size_t bracketPos = numStr.find('[');
        if (bracketPos != std::string::npos) {
            numStr = numStr.substr(0, bracketPos);
        }
        
        // Trim whitespace
        size_t start = numStr.find_first_not_of(" \t");
        size_t end = numStr.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            numStr = numStr.substr(start, end - start + 1);
        }
        
        // Try to parse as number first
        try {
            return std::stod(numStr);
        } catch (...) {
            // Not a number, try as variable name
        }
        
        // Try to get from variables
        try {
            return caseDef.getVariable(str);
        } catch (...) {
            return 0.0;
        }
    }
    
    static std::string joinIds(const std::set<int>& ids) {
        std::string result;
        for (int id : ids) {
            if (!result.empty()) result += ", ";
            result += std::to_string(id);
        }
        return result;
    }
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

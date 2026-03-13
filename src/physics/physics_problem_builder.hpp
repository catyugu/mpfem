#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "electrostatics_solver.hpp"
#include "heat_transfer_solver.hpp"
#include "coupling_manager.hpp"
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
    
    // Domain material mapping (for temperature-dependent properties)
    std::map<int, std::string> domainMaterial;
    
    // Solvers
    std::unique_ptr<ElectrostaticsSolver> electrostatics;
    std::unique_ptr<HeatTransferSolver> heatTransfer;
    // std::unique_ptr<SolidMechanicsSolver> solidMechanics;  // TODO
    
    // Coupling
    std::unique_ptr<CouplingManager> couplingManager;
    
    // Material coefficients (stored for coupling)
    std::unique_ptr<PWConstCoefficient> conductivity;
    std::unique_ptr<PWConstCoefficient> thermalConductivity;
    
    bool hasElectrostatics() const { return electrostatics != nullptr; }
    bool hasHeatTransfer() const { return heatTransfer != nullptr; }
    // bool hasSolidMechanics() const { return solidMechanics != nullptr; }
    
    bool hasJouleHeating() const {
        return hasElectrostatics() && hasHeatTransfer();
    }
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
        for (const auto& assign : caseDef.materialAssignments) {
            for (int domId : assign.domainIds) {
                setup.domainMaterial[domId] = assign.materialTag;
            }
        }
        
        // Find maximum domain ID
        int maxDomainId = 0;
        for (const auto& [domId, _] : setup.domainMaterial) {
            maxDomainId = std::max(maxDomainId, domId);
        }
        
        // Build each physics field
        for (const auto& physics : caseDef.physicsDefinitions) {
            if (physics.kind == "electrostatics") {
                buildElectrostatics(setup, physics, setup.domainMaterial, maxDomainId, materials);
            }
            else if (physics.kind == "heat_transfer") {
                buildHeatTransfer(setup, physics, setup.domainMaterial, maxDomainId, materials);
            }
            // else if (physics.kind == "solid_mechanics") {
            //     buildSolidMechanics(setup, physics, domainMaterial, maxDomainId, materials);
            // }
            else {
                LOG_WARN << "Unknown physics kind: " << physics.kind;
            }
        }
        
        // Build coupling manager if needed
        if (setup.hasJouleHeating()) {
            buildCouplingManager(setup, caseDef);
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
    
    static void buildHeatTransfer(
        PhysicsProblemSetup& setup,
        const PhysicsDefinition& physics,
        const std::map<int, std::string>& domainMaterial,
        int maxDomainId,
        const MaterialDatabase& materials)
    {
        LOG_INFO << "Building heat transfer solver, order = " << physics.order;
        
        // Create thermal conductivity coefficient
        auto thermalConductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
        
        for (const auto& [domId, matTag] : domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat) {
                double k = mat->thermalConductivity;
                thermalConductivity->setConstant(domId, k);
                LOG_INFO << "Domain " << domId << " (" << matTag << "): k = " << k;
            } else {
                LOG_WARN << "Material '" << matTag << "' not found for domain " << domId;
            }
        }
        
        // Create solver
        setup.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
        setup.heatTransfer->setSolverType(physics.solver.type);
        setup.heatTransfer->setMaxIterations(physics.solver.maxIterations);
        setup.heatTransfer->setTolerance(physics.solver.relativeTolerance);
        setup.heatTransfer->setPrintLevel(physics.solver.printLevel);
        
        // Initialize
        setup.heatTransfer->initialize(*setup.mesh, *thermalConductivity);
        
        // Store thermal conductivity for later use
        setup.thermalConductivity = std::move(thermalConductivity);
        
        // Apply boundary conditions
        for (const auto& bc : physics.boundaries) {
            if (bc.kind == "temperature") {
                // Dirichlet BC
                Real value = 0.0;
                auto it = bc.params.find("value");
                if (it != bc.params.end()) {
                    value = parseValue(it->second, setup.caseDef);
                }
                
                for (int id : bc.ids) {
                    setup.heatTransfer->addDirichletBC(id, value);
                    LOG_INFO << "BC: T = " << value << " K on boundary " << id;
                }
            }
            else if (bc.kind == "convection") {
                // Robin BC: h and Tinf
                Real h = 5.0, Tinf = 293.15;
                auto hit = bc.params.find("h");
                auto tit = bc.params.find("T_inf");
                if (hit != bc.params.end()) {
                    h = parseValue(hit->second, setup.caseDef);
                }
                if (tit != bc.params.end()) {
                    Tinf = parseValue(tit->second, setup.caseDef);
                }
                
                for (int id : bc.ids) {
                    setup.heatTransfer->addConvectionBC(id, h, Tinf);
                }
                LOG_INFO << "BC: convection h=" << h << " W/(m²K), Tinf=" << Tinf 
                         << " K on boundaries " << joinIds(bc.ids);
            }
            else if (bc.kind == "thermal_insulation") {
                // Natural BC (do nothing for Neumann zero)
                LOG_DEBUG << "BC: thermal insulation on boundaries " << joinIds(bc.ids);
            }
            else {
                LOG_WARN << "Unknown BC kind for heat transfer: " << bc.kind;
            }
        }
    }
    
    static void buildCouplingManager(PhysicsProblemSetup& setup, const CaseDefinition& caseDef) {
        // Parse coupling configuration
        IterationMethod method = IterationMethod::Picard;
        int maxIter = 15;
        Real tolerance = 1e-6;
        
        if (caseDef.couplingConfig.method == CouplingMethod::Picard) {
            method = IterationMethod::Picard;
        }
        maxIter = caseDef.couplingConfig.maxIterations;
        tolerance = caseDef.couplingConfig.tolerance;
        
        // Create temperature-dependent conductivity coefficient if needed
        if (setup.hasJouleHeating()) {
            int maxDomainId = 0;
            for (const auto& [domId, _] : setup.domainMaterial) {
                maxDomainId = std::max(maxDomainId, domId);
            }
            
            auto tempDepCond = std::make_unique<TemperatureDependentConductivityCoefficient>();
            
            // Prepare material arrays
            std::vector<Real> rho0(maxDomainId, 0.0);
            std::vector<Real> alpha(maxDomainId, 0.0);
            std::vector<Real> tref(maxDomainId, 293.15);
            std::vector<Real> sigma0(maxDomainId, 0.0);
            
            for (const auto& [domId, matTag] : setup.domainMaterial) {
                const MaterialPropertyModel* mat = setup.materials.getMaterial(matTag);
                if (mat) {
                    rho0[domId - 1] = mat->rho0;
                    alpha[domId - 1] = mat->alpha;
                    tref[domId - 1] = mat->tref;
                    sigma0[domId - 1] = mat->electricConductivity;
                    
                    LOG_DEBUG << "Temperature-dependent conductivity for domain " << domId 
                             << " (" << matTag << "): rho0=" << mat->rho0 
                             << ", alpha=" << mat->alpha << ", Tref=" << mat->tref;
                }
            }
            
            tempDepCond->setMaterialFields(rho0, alpha, tref, sigma0);
            setup.electrostatics->setTemperatureDependentConductivity(std::move(tempDepCond));
        }
        
        setup.couplingManager = std::make_unique<CouplingManager>(method, maxIter, tolerance);
        setup.couplingManager->setElectrostaticsSolver(setup.electrostatics.get());
        setup.couplingManager->setHeatTransferSolver(setup.heatTransfer.get());
        setup.couplingManager->enableJouleHeating(true);
        
        // Set temperature field for temperature-dependent conductivity
        setup.electrostatics->setTemperatureField(&setup.heatTransfer->field());
        
        LOG_INFO << "Coupling manager created: method=Picard"
                 << ", max_iter=" << maxIter << ", tol=" << tolerance;
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

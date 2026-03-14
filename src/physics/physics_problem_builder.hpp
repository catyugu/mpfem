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

struct PhysicsProblemSetup {
    std::string caseName;
    std::unique_ptr<Mesh> mesh;
    MaterialDatabase materials;
    CaseDefinition caseDef;
    std::map<int, std::string> domainMaterial;
    
    std::unique_ptr<ElectrostaticsSolver> electrostatics;
    std::unique_ptr<HeatTransferSolver> heatTransfer;
    std::unique_ptr<CouplingManager> couplingManager;
    
    std::unique_ptr<PWConstCoefficient> conductivity;
    std::unique_ptr<PWConstCoefficient> thermalConductivity;
    
    bool hasElectrostatics() const { return electrostatics != nullptr; }
    bool hasHeatTransfer() const { return heatTransfer != nullptr; }
    bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
};

class PhysicsProblemBuilder {
public:
    static PhysicsProblemSetup build(const std::string& caseDir) {
        PhysicsProblemSetup setup;
        
        std::string casePath = caseDir + "/case.xml";
        LOG_INFO << "Reading case from " << casePath;
        
        CaseXmlReader::readFromFile(casePath, setup.caseDef);
        setup.caseName = setup.caseDef.caseName;
        
        std::string meshPath = caseDir + "/" + setup.caseDef.meshPath;
        LOG_INFO << "Reading mesh from " << meshPath;
        
        setup.mesh = std::make_unique<Mesh>(MphtxtReader::read(meshPath));
        LOG_INFO << "Mesh loaded: " << setup.mesh->numVertices() << " vertices, "
                 << setup.mesh->numElements() << " elements";
        
        std::string matPath = caseDir + "/" + setup.caseDef.materialsPath;
        LOG_INFO << "Reading materials from " << matPath;
        
        MaterialXmlReader::readFromFile(matPath, setup.materials);
        
        buildSolvers(setup);
        
        return setup;
    }
    
private:
    static void buildSolvers(PhysicsProblemSetup& setup) {
        const auto& caseDef = setup.caseDef;
        const auto& materials = setup.materials;
        
        // Build domain material mapping
        for (const auto& assign : caseDef.materialAssignments) {
            for (int domId : assign.domainIds) {
                setup.domainMaterial[domId] = assign.materialTag;
            }
        }
        
        int maxDomainId = 0;
        for (const auto& [domId, _] : setup.domainMaterial) {
            maxDomainId = std::max(maxDomainId, domId);
        }
        
        // Build physics solvers
        for (const auto& physics : caseDef.physicsDefinitions) {
            if (physics.kind == "electrostatics") {
                buildElectrostatics(setup, physics, maxDomainId, materials);
            }
            else if (physics.kind == "heat_transfer") {
                buildHeatTransfer(setup, physics, maxDomainId, materials);
            }
        }
        
        // Setup coupling if needed
        if (setup.hasJouleHeating()) {
            setup.couplingManager = std::make_unique<CouplingManager>();
            setup.couplingManager->setElectrostaticsSolver(setup.electrostatics.get());
            setup.couplingManager->setHeatTransferSolver(setup.heatTransfer.get());
            setup.couplingManager->setTolerance(caseDef.couplingConfig.tolerance);
            setup.couplingManager->setMaxIterations(caseDef.couplingConfig.maxIterations);
            
            // Setup Joule heating domains from coupledPhysicsDefinitions
            for (const auto& cp : caseDef.coupledPhysicsDefinitions) {
                if (cp.kind == "joule_heating") {
                    setup.couplingManager->setJouleHeatDomains(cp.domainIds);
                    LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
                }
            }
            
            // Setup temperature-dependent conductivity
            setupCoupling(setup, materials);
        }
    }
    
    static void buildElectrostatics(
        PhysicsProblemSetup& setup,
        const PhysicsDefinition& physics,
        int maxDomainId,
        const MaterialDatabase& materials)
    {
        LOG_INFO << "Building electrostatics solver, order = " << physics.order;
        
        auto conductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
        
        // Set conductivity values
        for (const auto& [domId, matTag] : setup.domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat) {
                conductivity->set(domId, mat->electricConductivity);
                LOG_INFO << "Domain " << domId << " (" << matTag 
                         << "): sigma = " << mat->electricConductivity;
            }
        }
        
        setup.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
        setup.electrostatics->setSolver(physics.solver.type, physics.solver.maxIterations, physics.solver.relativeTolerance);
        setup.electrostatics->initialize(*setup.mesh, *conductivity);
        
        // Apply boundary conditions
        for (const auto& bc : physics.boundaries) {
            if (bc.kind == "voltage") {
                Real value = parseValue(bc.params, "value", setup.caseDef);
                for (int id : bc.ids) {
                    setup.electrostatics->addDirichletBC(id, value);
                }
            }
        }
        
        setup.conductivity = std::move(conductivity);
    }
    
    static void buildHeatTransfer(
        PhysicsProblemSetup& setup,
        const PhysicsDefinition& physics,
        int maxDomainId,
        const MaterialDatabase& materials)
    {
        LOG_INFO << "Building heat transfer solver, order = " << physics.order;
        
        auto thermalConductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
        
        for (const auto& [domId, matTag] : setup.domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat) {
                thermalConductivity->set(domId, mat->thermalConductivity);
            }
        }
        
        setup.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
        setup.heatTransfer->setSolver(physics.solver.type, physics.solver.maxIterations, physics.solver.relativeTolerance);
        setup.heatTransfer->initialize(*setup.mesh, *thermalConductivity);
        
        for (const auto& bc : physics.boundaries) {
            if (bc.kind == "temperature") {
                Real value = parseValue(bc.params, "value", setup.caseDef);
                for (int id : bc.ids) {
                    setup.heatTransfer->addDirichletBC(id, value);
                }
            }
            else if (bc.kind == "convection") {
                Real h = parseValue(bc.params, "h", setup.caseDef, 5.0);
                Real Tinf = parseValue(bc.params, "T_inf", setup.caseDef, 293.15);
                for (int id : bc.ids) {
                    setup.heatTransfer->addConvectionBC(id, h, Tinf);
                }
            }
        }
        
        setup.thermalConductivity = std::move(thermalConductivity);
    }
    
    static void setupCoupling(PhysicsProblemSetup& setup, const MaterialDatabase& materials) {
        bool hasTempDepSigma = false;
        
        // Check if any material has temperature-dependent resistivity
        for (const auto& [domId, matTag] : setup.domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat && mat->rho0 > 0.0) {
                hasTempDepSigma = true;
                break;
            }
        }
        
        if (!hasTempDepSigma) return;
        
        // Setup temperature-dependent conductivity
        setup.couplingManager->enableTempDependentConductivity();
        
        for (const auto& [domId, matTag] : setup.domainMaterial) {
            const MaterialPropertyModel* mat = materials.getMaterial(matTag);
            if (mat) {
                if (mat->rho0 > 0.0) {
                    LOG_INFO << "Domain " << domId << " (" << matTag 
                             << "): temp-dep sigma, rho0 = " << mat->rho0 
                             << ", alpha = " << mat->alpha;
                    setup.couplingManager->setTempDepMaterial(
                        domId, mat->rho0, mat->alpha, mat->tref);
                } else {
                    setup.couplingManager->setConstantConductivity(
                        domId, mat->electricConductivity);
                }
            }
        }
    }
    
    static Real parseValue(const std::map<std::string, std::string>& params, 
                           const std::string& key,
                           const CaseDefinition& caseDef,
                           Real defaultVal = 0.0) {
        auto it = params.find(key);
        if (it == params.end()) return defaultVal;
        
        std::string numStr = it->second;
        size_t bracketPos = numStr.find('[');
        if (bracketPos != std::string::npos) {
            numStr = numStr.substr(0, bracketPos);
        }
        
        try {
            return std::stod(numStr);
        } catch (...) {
            try {
                return caseDef.getVariable(numStr);
            } catch (...) {
                return defaultVal;
            }
        }
    }
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

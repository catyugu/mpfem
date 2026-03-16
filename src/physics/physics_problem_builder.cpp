#include "physics_problem_builder.hpp"

namespace mpfem {

PhysicsProblemSetup PhysicsProblemBuilder::build(const std::string& caseDir) {
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

void PhysicsProblemBuilder::buildSolvers(PhysicsProblemSetup& setup) {
    const auto& caseDef = setup.caseDef;
    const auto& materials = setup.materials;
    
    // 构建域材料映射
    for (const auto& assign : caseDef.materialAssignments) {
        for (int domId : assign.domainIds) {
            setup.domainMaterial[domId] = assign.materialTag;
        }
    }
    
    int maxDomainId = 0;
    for (const auto& [domId, _] : setup.domainMaterial) {
        maxDomainId = std::max(maxDomainId, domId);
    }
    
    // 构建物理场求解器
    for (const auto& physics : caseDef.physicsDefinitions) {
        if (physics.kind == "electrostatics") {
            buildElectrostatics(setup, physics, maxDomainId);
        }
        else if (physics.kind == "heat_transfer") {
            buildHeatTransfer(setup, physics, maxDomainId);
        }
        else if (physics.kind == "solid_mechanics") {
            buildStructural(setup, physics, maxDomainId);
        }
    }
    
    // 设置耦合
    if (setup.hasJouleHeating() || setup.hasThermalExpansion()) {
        setup.couplingManager = std::make_unique<CouplingManager>();
        
        if (setup.hasElectrostatics()) {
            setup.couplingManager->setElectrostaticsSolver(setup.electrostatics.get());
        }
        if (setup.hasHeatTransfer()) {
            setup.couplingManager->setHeatTransferSolver(setup.heatTransfer.get());
        }
        if (setup.hasStructural()) {
            setup.couplingManager->setStructuralSolver(setup.structural.get());
            // 设置结构场材料参数用于热膨胀计算
            setup.couplingManager->setStructuralMaterial(
                setup.youngModulus.get(), setup.poissonRatio.get());
        }
        
        setup.couplingManager->setTolerance(caseDef.couplingConfig.tolerance);
        setup.couplingManager->setMaxIterations(caseDef.couplingConfig.maxIterations);
        
        // 设置焦耳热耦合
        for (const auto& cp : caseDef.coupledPhysicsDefinitions) {
            if (cp.kind == "joule_heating") {
                setup.couplingManager->setJouleHeatDomains(cp.domainIds);
                LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
            }
            else if (cp.kind == "thermal_expansion") {
                // 设置热膨胀耦合
                for (int domId : cp.domainIds) {
                    const MaterialPropertyModel* mat = materials.getMaterial(setup.domainMaterial[domId]);
                    if (mat && mat->thermalExpansion > 0.0) {
                        setup.couplingManager->setThermalExpansion(domId, mat->thermalExpansion, 293.15);
                    }
                }
                LOG_INFO << "Thermal expansion coupling enabled";
            }
        }
        
        // 设置温度依赖电导率
        setupCoupling(setup);
    }
}

void PhysicsProblemBuilder::buildElectrostatics(
    PhysicsProblemSetup& setup,
    const PhysicsDefinition& physics,
    int maxDomainId)
{
    const auto& materials = setup.materials;
    LOG_INFO << "Building electrostatics solver, order = " << physics.order;
    
    // 创建电导率系数
    auto conductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
    
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            conductivity->set(domId, mat->electricConductivity);
            LOG_INFO << "Domain " << domId << " (" << matTag 
                     << "): sigma = " << mat->electricConductivity;
        }
    }
    
    // 创建并初始化求解器
    setup.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
    setup.electrostatics->setSolverConfig(physics.solver);
    setup.electrostatics->initialize(*setup.mesh);
    setup.electrostatics->setConductivity(conductivity.get());
    
    // 应用电压边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "voltage") {
            Real value = parseValue(bc.params, "value", setup.caseDef);
            for (int id : bc.ids) {
                setup.electrostatics->addVoltageBC(id, value);
            }
        }
    }
    
    setup.conductivity = std::move(conductivity);
}

void PhysicsProblemBuilder::buildHeatTransfer(
    PhysicsProblemSetup& setup,
    const PhysicsDefinition& physics,
    int maxDomainId)
{
    const auto& materials = setup.materials;
    LOG_INFO << "Building heat transfer solver, order = " << physics.order;
    
    // 创建热导率系数
    auto thermalConductivity = std::make_unique<PWConstCoefficient>(maxDomainId);
    
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            thermalConductivity->set(domId, mat->thermalConductivity);
        }
    }
    
    // 创建并初始化求解器
    setup.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
    setup.heatTransfer->setSolverConfig(physics.solver);
    setup.heatTransfer->initialize(*setup.mesh);
    setup.heatTransfer->setConductivity(thermalConductivity.get());
    
    // 应用边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "temperature") {
            Real value = parseValue(bc.params, "value", setup.caseDef);
            for (int id : bc.ids) {
                setup.heatTransfer->addTemperatureBC(id, value);
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

void PhysicsProblemBuilder::buildStructural(
    PhysicsProblemSetup& setup,
    const PhysicsDefinition& physics,
    int maxDomainId)
{
    const auto& materials = setup.materials;
    LOG_INFO << "Building structural solver, order = " << physics.order;
    
    // 创建材料系数
    auto youngModulus = std::make_unique<PWConstCoefficient>(maxDomainId);
    auto poissonRatio = std::make_unique<PWConstCoefficient>(maxDomainId);
    auto thermalExpansion = std::make_unique<PWConstCoefficient>(maxDomainId);
    
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            youngModulus->set(domId, mat->youngModulus);
            poissonRatio->set(domId, mat->poissonRatio);
            thermalExpansion->set(domId, mat->thermalExpansion);
            LOG_INFO << "Domain " << domId << " (" << matTag 
                     << "): E = " << mat->youngModulus 
                     << ", nu = " << mat->poissonRatio
                     << ", alpha_T = " << mat->thermalExpansion;
        }
    }
    
    // 创建并初始化求解器
    setup.structural = std::make_unique<StructuralSolver>(physics.order);
    setup.structural->setSolverConfig(physics.solver);
    setup.structural->initialize(*setup.mesh);
    setup.structural->setMaterial(youngModulus.get(), poissonRatio.get());
    
    // 应用边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "fixed_constraint") {
            for (int id : bc.ids) {
                setup.structural->addFixedDisplacementBC(id, Vector3(0.0, 0.0, 0.0));
            }
        }
    }
    
    setup.youngModulus = std::move(youngModulus);
    setup.poissonRatio = std::move(poissonRatio);
    setup.thermalExpansion = std::move(thermalExpansion);
}

void PhysicsProblemBuilder::setupCoupling(PhysicsProblemSetup& setup) {
    const auto& materials = setup.materials;
    bool hasTempDepSigma = false;
    
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat && mat->rho0 > 0.0) {
            hasTempDepSigma = true;
            break;
        }
    }
    
    if (!hasTempDepSigma) return;
    
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

Real PhysicsProblemBuilder::parseValue(const std::map<std::string, std::string>& params, 
                                        const std::string& key,
                                        const CaseDefinition& caseDef,
                                        Real defaultVal) {
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

}  // namespace mpfem

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
        }
        
        setup.couplingManager->setTolerance(caseDef.couplingConfig.tolerance);
        setup.couplingManager->setMaxIterations(caseDef.couplingConfig.maxIterations);
        
        // 设置焦耳热耦合
        for (const auto& cp : caseDef.coupledPhysicsDefinitions) {
            if (cp.kind == "joule_heating") {
                // 创建焦耳热系数
                auto* jouleHeat = setup.couplingManager->createCoefficientNamed<JouleHeatCoefficient>("joule_heat");
                std::set<int> domains(cp.domainIds.begin(), cp.domainIds.end());
                jouleHeat->setDomains(domains);
                LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
            }
            else if (cp.kind == "thermal_expansion") {
                // 创建热膨胀系数
                auto* thermalExp = setup.couplingManager->createCoefficientNamed<ThermalExpansionCoefficient>("thermal_expansion");
                thermalExp->setReferenceTemperature(293.15);
                
                for (int domId : cp.domainIds) {
                    const MaterialPropertyModel* mat = materials.getMaterial(setup.domainMaterial[domId]);
                    if (mat && mat->thermalExpansion > 0.0) {
                        thermalExp->setAlphaT(domId, mat->thermalExpansion);
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
    
    // 创建并初始化求解器
    setup.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
    setup.electrostatics->setSolverConfig(physics.solver);
    setup.electrostatics->initialize(*setup.mesh);
    
    // 设置每个域的电导率
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            // 为每个域创建单独的常量系数
            std::string key = "conductivity_" + std::to_string(domId);
            setup.bcCoefficients[key] = std::make_unique<ConstantCoefficient>(mat->electricConductivity);
            setup.electrostatics->setConductivity({domId}, setup.bcCoefficients[key].get());
            LOG_INFO << "Domain " << domId << " (" << matTag 
                     << "): sigma = " << mat->electricConductivity;
        }
    }
    
    // 应用电压边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "voltage") {
            Real value = parseValue(bc.params, "value", setup.caseDef);
            std::string key = "voltage_bc_" + std::to_string(*bc.ids.begin());
            setup.bcCoefficients[key] = std::make_unique<ConstantCoefficient>(value);
            std::set<int> ids(bc.ids.begin(), bc.ids.end());
            setup.electrostatics->addVoltageBC(ids, setup.bcCoefficients[key].get());
        }
    }
}

void PhysicsProblemBuilder::buildHeatTransfer(
    PhysicsProblemSetup& setup,
    const PhysicsDefinition& physics,
    int maxDomainId)
{
    const auto& materials = setup.materials;
    LOG_INFO << "Building heat transfer solver, order = " << physics.order;
    
    // 创建并初始化求解器
    setup.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
    setup.heatTransfer->setSolverConfig(physics.solver);
    setup.heatTransfer->initialize(*setup.mesh);
    
    // 设置每个域的热导率
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            std::string key = "thermal_conductivity_" + std::to_string(domId);
            setup.bcCoefficients[key] = std::make_unique<ConstantCoefficient>(mat->thermalConductivity);
            setup.heatTransfer->setConductivity({domId}, setup.bcCoefficients[key].get());
        }
    }
    
    // 应用边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "temperature") {
            Real value = parseValue(bc.params, "value", setup.caseDef);
            std::string key = "temp_bc_" + std::to_string(*bc.ids.begin());
            setup.bcCoefficients[key] = std::make_unique<ConstantCoefficient>(value);
            std::set<int> ids(bc.ids.begin(), bc.ids.end());
            setup.heatTransfer->addTemperatureBC(ids, setup.bcCoefficients[key].get());
        }
        else if (bc.kind == "convection") {
            Real h = parseValue(bc.params, "h", setup.caseDef, 5.0);
            Real Tinf = parseValue(bc.params, "T_inf", setup.caseDef, 293.15);
            
            std::string hKey = "conv_h_" + std::to_string(*bc.ids.begin());
            std::string tinfKey = "conv_tinf_" + std::to_string(*bc.ids.begin());
            setup.bcCoefficients[hKey] = std::make_unique<ConstantCoefficient>(h);
            setup.bcCoefficients[tinfKey] = std::make_unique<ConstantCoefficient>(Tinf);
            
            std::set<int> ids(bc.ids.begin(), bc.ids.end());
            setup.heatTransfer->addConvectionBC(ids, 
                setup.bcCoefficients[hKey].get(), 
                setup.bcCoefficients[tinfKey].get());
        }
    }
}

void PhysicsProblemBuilder::buildStructural(
    PhysicsProblemSetup& setup,
    const PhysicsDefinition& physics,
    int maxDomainId)
{
    const auto& materials = setup.materials;
    LOG_INFO << "Building structural solver, order = " << physics.order;
    
    // 创建并初始化求解器
    setup.structural = std::make_unique<StructuralSolver>(physics.order);
    setup.structural->setSolverConfig(physics.solver);
    setup.structural->initialize(*setup.mesh);
    
    // 设置每个域的材料参数
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            std::string eKey = "young_" + std::to_string(domId);
            std::string nuKey = "poisson_" + std::to_string(domId);
            setup.bcCoefficients[eKey] = std::make_unique<ConstantCoefficient>(mat->youngModulus);
            setup.bcCoefficients[nuKey] = std::make_unique<ConstantCoefficient>(mat->poissonRatio);
            
            setup.structural->setYoungModulus({domId}, setup.bcCoefficients[eKey].get());
            setup.structural->setPoissonRatio({domId}, setup.bcCoefficients[nuKey].get());
            
            LOG_INFO << "Domain " << domId << " (" << matTag 
                     << "): E = " << mat->youngModulus 
                     << ", nu = " << mat->poissonRatio;
        }
    }
    
    // 应用边界条件
    for (const auto& bc : physics.boundaries) {
        if (bc.kind == "fixed_constraint") {
            std::string key = "fixed_disp_" + std::to_string(*bc.ids.begin());
            setup.bcVectorCoefficients[key] = std::make_unique<ConstantVectorCoefficient>(0.0, 0.0, 0.0);
            std::set<int> ids(bc.ids.begin(), bc.ids.end());
            setup.structural->addFixedDisplacementBC(ids, setup.bcVectorCoefficients[key].get());
        }
    }
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
    
    // 创建温度依赖电导率系数
    auto* tempDepSigma = setup.couplingManager->createCoefficientNamed<TemperatureDependentConductivity>("temp_dep_sigma");
    
    for (const auto& [domId, matTag] : setup.domainMaterial) {
        const MaterialPropertyModel* mat = materials.getMaterial(matTag);
        if (mat) {
            if (mat->rho0 > 0.0) {
                LOG_INFO << "Domain " << domId << " (" << matTag 
                         << "): temp-dep sigma, rho0 = " << mat->rho0 
                         << ", alpha = " << mat->alpha;
                tempDepSigma->setMaterial(domId, mat->rho0, mat->alpha, mat->tref);
            } else {
                tempDepSigma->setConstantConductivity(domId, mat->electricConductivity);
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

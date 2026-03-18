#include "physics_problem_builder.hpp"

namespace mpfem
{

    std::unique_ptr<SteadyProblem> PhysicsProblemBuilder::build(const std::string &caseDir)
    {
        auto problem = std::make_unique<SteadyProblem>();

        std::string casePath = caseDir + "/case.xml";
        LOG_INFO << "Reading case from " << casePath;

        CaseXmlReader::readFromFile(casePath, problem->caseDef);
        problem->caseName = problem->caseDef.caseName;

        std::string meshPath = caseDir + "/" + problem->caseDef.meshPath;
        LOG_INFO << "Reading mesh from " << meshPath;

        problem->mesh = std::make_unique<Mesh>(MphtxtReader::read(meshPath));
        LOG_INFO << "Mesh loaded: " << problem->mesh->numVertices() << " vertices, "
                 << problem->mesh->numElements() << " elements";

        std::string matPath = caseDir + "/" + problem->caseDef.materialsPath;
        LOG_INFO << "Reading materials from " << matPath;

        MaterialXmlReader::readFromFile(matPath, problem->materials);

        buildSolvers(*problem);

        return problem;
    }

    void PhysicsProblemBuilder::buildSolvers(SteadyProblem &problem)
    {
        const auto &caseDef = problem.caseDef;
        for (const auto &assign : caseDef.materialAssignments)
            for (int domId : assign.domainIds)
                problem.domainMaterial[domId] = assign.materialTag;

        for (const auto &physics : caseDef.physicsDefinitions)
        {
            if (physics.kind == "electrostatics") buildElectrostatics(problem, physics);
            else if (physics.kind == "heat_transfer") buildHeatTransfer(problem, physics);
            else if (physics.kind == "solid_mechanics") buildStructural(problem, physics);
        }

        if (problem.hasJouleHeating() || problem.hasThermalExpansion())
        {
            problem.couplingMaxIter = caseDef.couplingConfig.maxIterations;
            problem.couplingTol = caseDef.couplingConfig.tolerance;
            setupCoupling(problem);
        }
    }

    void PhysicsProblemBuilder::buildElectrostatics(SteadyProblem &problem, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building electrostatics solver, order = " << physics.order;
        problem.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
        problem.electrostatics->setSolverConfig(physics.solver);
        problem.electrostatics->initialize(*problem.mesh);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            if (const auto* mat = problem.materials.getMaterial(matTag))
            {
                std::string key = "conductivity_" + std::to_string(domId);
                problem.setCoef(key, std::make_unique<ConstantCoefficient>(mat->electricConductivity));
                problem.electrostatics->setConductivity({domId}, problem.getCoef<Coefficient>(key));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "voltage")
            {
                std::string key = "voltage_bc_" + std::to_string(*bc.ids.begin());
                problem.setCoef(key, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "value", problem.caseDef)));
                problem.electrostatics->addVoltageBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<Coefficient>(key));
            }
        }
    }

    void PhysicsProblemBuilder::buildHeatTransfer(SteadyProblem &problem, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building heat transfer solver, order = " << physics.order;
        problem.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
        problem.heatTransfer->setSolverConfig(physics.solver);
        problem.heatTransfer->initialize(*problem.mesh);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            if (const auto* mat = problem.materials.getMaterial(matTag))
            {
                std::string key = "thermal_conductivity_" + std::to_string(domId);
                problem.setCoef(key, std::make_unique<ConstantCoefficient>(mat->thermalConductivity));
                problem.heatTransfer->setConductivity({domId}, problem.getCoef<Coefficient>(key));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "temperature")
            {
                std::string key = "temp_bc_" + std::to_string(*bc.ids.begin());
                problem.setCoef(key, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "value", problem.caseDef)));
                problem.heatTransfer->addTemperatureBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<Coefficient>(key));
            }
            else if (bc.kind == "convection")
            {
                std::string hKey = "conv_h_" + std::to_string(*bc.ids.begin());
                std::string tinfKey = "conv_tinf_" + std::to_string(*bc.ids.begin());
                problem.setCoef(hKey, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "h", problem.caseDef, 5.0)));
                problem.setCoef(tinfKey, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "T_inf", problem.caseDef, 293.15)));
                problem.heatTransfer->addConvectionBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<Coefficient>(hKey), problem.getCoef<Coefficient>(tinfKey));
            }
        }
    }

    void PhysicsProblemBuilder::buildStructural(SteadyProblem &problem, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building structural solver, order = " << physics.order;
        problem.structural = std::make_unique<StructuralSolver>(physics.order);
        problem.structural->setSolverConfig(physics.solver);
        problem.structural->initialize(*problem.mesh);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            if (const auto* mat = problem.materials.getMaterial(matTag))
            {
                std::string eKey = "young_" + std::to_string(domId);
                std::string nuKey = "poisson_" + std::to_string(domId);
                problem.setCoef(eKey, std::make_unique<ConstantCoefficient>(mat->youngModulus));
                problem.setCoef(nuKey, std::make_unique<ConstantCoefficient>(mat->poissonRatio));
                problem.structural->setYoungModulus({domId}, problem.getCoef<Coefficient>(eKey));
                problem.structural->setPoissonRatio({domId}, problem.getCoef<Coefficient>(nuKey));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "fixed_constraint")
            {
                std::string key = "fixed_disp_" + std::to_string(*bc.ids.begin());
                problem.setCoef(key, std::make_unique<ConstantVectorCoefficient>(0.0, 0.0, 0.0));
                problem.structural->addFixedDisplacementBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<VectorCoefficient>(key));
            }
        }
    }

    void PhysicsProblemBuilder::setupCoupling(SteadyProblem &problem)
    {
        DomainMappedCoefficient tempDepSigmaMap;
        
        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            if (const auto* mat = problem.materials.getMaterial(matTag))
            {
                std::string key = "tempDepSigma_" + std::to_string(domId);
                if (mat->rho0 > 0.0)
                {
                    problem.setCoef(key, std::make_unique<TemperatureDependentConductivity>(
                        problem.heatTransfer->field(), mat->rho0, mat->alpha, mat->tref));
                    tempDepSigmaMap.set(domId, problem.getCoef<Coefficient>(key));
                }
                else if (auto* c = problem.getCoef<Coefficient>("conductivity_" + std::to_string(domId)))
                {
                    tempDepSigmaMap.set(domId, c);
                }
            }
        }
        
        problem.setCoef("tempDepSigmaMap", std::make_unique<DomainMappedCoefficient>(std::move(tempDepSigmaMap)));
        problem.electrostatics->setConductivity(problem.getCoef<Coefficient>("tempDepSigmaMap"));
        
        for (const auto &cp : problem.caseDef.coupledPhysicsDefinitions)
        {
            if (cp.kind == "joule_heating")
            {
                problem.setCoef("jouleHeat", std::make_unique<JouleHeatCoefficient>(
                    problem.electrostatics->field(), problem.electrostatics->conductivity()));
                
                DomainMappedCoefficient jouleHeatMap;
                jouleHeatMap.set({cp.domainIds.begin(), cp.domainIds.end()}, problem.getCoef<Coefficient>("jouleHeat"));
                problem.setCoef("jouleHeatMap", std::make_unique<DomainMappedCoefficient>(std::move(jouleHeatMap)));
                problem.heatTransfer->setHeatSource(problem.getCoef<Coefficient>("jouleHeatMap"));
                LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
            }
            else if (cp.kind == "thermal_expansion")
            {
                for (int domId : cp.domainIds)
                {
                    if (const auto* mat = problem.materials.getMaterial(problem.domainMaterial[domId]); mat && mat->thermalExpansion > 0.0)
                    {
                        std::string key = "thermalExp_" + std::to_string(domId);
                        problem.setCoef(key, std::make_unique<ThermalExpansionCoefficient>(
                            problem.heatTransfer->field(), mat->thermalExpansion, 293.15));
                        problem.structural->setThermalExpansion({domId}, problem.getCoef<Coefficient>(key));
                    }
                }
                LOG_INFO << "Thermal expansion coupling enabled";
            }
        }
    }

    Real PhysicsProblemBuilder::parseValue(const std::map<std::string, std::string> &params,
                                           const std::string &key, const CaseDefinition &caseDef, Real defaultVal)
    {
        auto it = params.find(key);
        if (it == params.end()) return defaultVal;
        std::string numStr = it->second;
        if (auto pos = numStr.find('['); pos != std::string::npos) numStr = numStr.substr(0, pos);
        try { return std::stod(numStr); }
        catch (...) { try { return caseDef.getVariable(numStr); } catch (...) { return defaultVal; } }
    }

} // namespace mpfem

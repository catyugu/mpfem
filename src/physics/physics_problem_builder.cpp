#include "physics_problem_builder.hpp"

namespace mpfem
{

    PhysicsProblemSetup PhysicsProblemBuilder::build(const std::string &caseDir)
    {
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

    void PhysicsProblemBuilder::buildSolvers(PhysicsProblemSetup &setup)
    {
        const auto &caseDef = setup.caseDef;
        for (const auto &assign : caseDef.materialAssignments)
            for (int domId : assign.domainIds)
                setup.domainMaterial[domId] = assign.materialTag;

        for (const auto &physics : caseDef.physicsDefinitions)
        {
            if (physics.kind == "electrostatics") buildElectrostatics(setup, physics);
            else if (physics.kind == "heat_transfer") buildHeatTransfer(setup, physics);
            else if (physics.kind == "solid_mechanics") buildStructural(setup, physics);
        }

        if (setup.hasJouleHeating() || setup.hasThermalExpansion())
        {
            setup.couplingMaxIter_ = caseDef.couplingConfig.maxIterations;
            setup.couplingTol_ = caseDef.couplingConfig.tolerance;
            setupCoupling(setup);
        }
    }

    void PhysicsProblemBuilder::buildElectrostatics(PhysicsProblemSetup &setup, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building electrostatics solver, order = " << physics.order;
        setup.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
        setup.electrostatics->setSolverConfig(physics.solver);
        setup.electrostatics->initialize(*setup.mesh);

        for (const auto &[domId, matTag] : setup.domainMaterial)
        {
            if (const auto* mat = setup.materials.getMaterial(matTag))
            {
                std::string key = "conductivity_" + std::to_string(domId);
                setup.set(key, std::make_unique<ConstantCoefficient>(mat->electricConductivity));
                setup.electrostatics->setConductivity({domId}, setup.get<Coefficient>(key));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "voltage")
            {
                std::string key = "voltage_bc_" + std::to_string(*bc.ids.begin());
                setup.set(key, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "value", setup.caseDef)));
                setup.electrostatics->addVoltageBC({bc.ids.begin(), bc.ids.end()}, setup.get<Coefficient>(key));
            }
        }
    }

    void PhysicsProblemBuilder::buildHeatTransfer(PhysicsProblemSetup &setup, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building heat transfer solver, order = " << physics.order;
        setup.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
        setup.heatTransfer->setSolverConfig(physics.solver);
        setup.heatTransfer->initialize(*setup.mesh);

        for (const auto &[domId, matTag] : setup.domainMaterial)
        {
            if (const auto* mat = setup.materials.getMaterial(matTag))
            {
                std::string key = "thermal_conductivity_" + std::to_string(domId);
                setup.set(key, std::make_unique<ConstantCoefficient>(mat->thermalConductivity));
                setup.heatTransfer->setConductivity({domId}, setup.get<Coefficient>(key));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "temperature")
            {
                std::string key = "temp_bc_" + std::to_string(*bc.ids.begin());
                setup.set(key, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "value", setup.caseDef)));
                setup.heatTransfer->addTemperatureBC({bc.ids.begin(), bc.ids.end()}, setup.get<Coefficient>(key));
            }
            else if (bc.kind == "convection")
            {
                std::string hKey = "conv_h_" + std::to_string(*bc.ids.begin());
                std::string tinfKey = "conv_tinf_" + std::to_string(*bc.ids.begin());
                setup.set(hKey, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "h", setup.caseDef, 5.0)));
                setup.set(tinfKey, std::make_unique<ConstantCoefficient>(parseValue(bc.params, "T_inf", setup.caseDef, 293.15)));
                setup.heatTransfer->addConvectionBC({bc.ids.begin(), bc.ids.end()}, setup.get<Coefficient>(hKey), setup.get<Coefficient>(tinfKey));
            }
        }
    }

    void PhysicsProblemBuilder::buildStructural(PhysicsProblemSetup &setup, const PhysicsDefinition &physics)
    {
        LOG_INFO << "Building structural solver, order = " << physics.order;
        setup.structural = std::make_unique<StructuralSolver>(physics.order);
        setup.structural->setSolverConfig(physics.solver);
        setup.structural->initialize(*setup.mesh);

        for (const auto &[domId, matTag] : setup.domainMaterial)
        {
            if (const auto* mat = setup.materials.getMaterial(matTag))
            {
                std::string eKey = "young_" + std::to_string(domId);
                std::string nuKey = "poisson_" + std::to_string(domId);
                setup.set(eKey, std::make_unique<ConstantCoefficient>(mat->youngModulus));
                setup.set(nuKey, std::make_unique<ConstantCoefficient>(mat->poissonRatio));
                setup.structural->setYoungModulus({domId}, setup.get<Coefficient>(eKey));
                setup.structural->setPoissonRatio({domId}, setup.get<Coefficient>(nuKey));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind == "fixed_constraint")
            {
                std::string key = "fixed_disp_" + std::to_string(*bc.ids.begin());
                setup.set(key, std::make_unique<ConstantVectorCoefficient>(0.0, 0.0, 0.0));
                setup.structural->addFixedDisplacementBC({bc.ids.begin(), bc.ids.end()}, setup.get<VectorCoefficient>(key));
            }
        }
    }

    void PhysicsProblemBuilder::setupCoupling(PhysicsProblemSetup &setup)
    {
        DomainMappedCoefficient tempDepSigmaMap;
        
        for (const auto &[domId, matTag] : setup.domainMaterial)
        {
            if (const auto* mat = setup.materials.getMaterial(matTag))
            {
                std::string key = "tempDepSigma_" + std::to_string(domId);
                if (mat->rho0 > 0.0)
                {
                    setup.set(key, std::make_unique<TemperatureDependentConductivity>(
                        setup.heatTransfer->field(), mat->rho0, mat->alpha, mat->tref));
                    tempDepSigmaMap.set(domId, setup.get<Coefficient>(key));
                }
                else if (auto* c = setup.get<Coefficient>("conductivity_" + std::to_string(domId)))
                {
                    tempDepSigmaMap.set(domId, c);
                }
            }
        }
        
        setup.set("tempDepSigmaMap", std::make_unique<DomainMappedCoefficient>(std::move(tempDepSigmaMap)));
        setup.electrostatics->setConductivity(setup.get<Coefficient>("tempDepSigmaMap"));
        
        for (const auto &cp : setup.caseDef.coupledPhysicsDefinitions)
        {
            if (cp.kind == "joule_heating")
            {
                setup.set("jouleHeat", std::make_unique<JouleHeatCoefficient>(
                    setup.electrostatics->field(), setup.electrostatics->conductivity()));
                
                DomainMappedCoefficient jouleHeatMap;
                jouleHeatMap.set({cp.domainIds.begin(), cp.domainIds.end()}, setup.get<Coefficient>("jouleHeat"));
                setup.set("jouleHeatMap", std::make_unique<DomainMappedCoefficient>(std::move(jouleHeatMap)));
                setup.heatTransfer->setHeatSource(setup.get<Coefficient>("jouleHeatMap"));
                LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
            }
            else if (cp.kind == "thermal_expansion")
            {
                for (int domId : cp.domainIds)
                {
                    if (const auto* mat = setup.materials.getMaterial(setup.domainMaterial[domId]); mat && mat->thermalExpansion > 0.0)
                    {
                        std::string key = "thermalExp_" + std::to_string(domId);
                        setup.set(key, std::make_unique<ThermalExpansionCoefficient>(
                            setup.heatTransfer->field(), mat->thermalExpansion, 293.15));
                        setup.structural->setThermalExpansion({domId}, setup.get<Coefficient>(key));
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

    CouplingResult PhysicsProblemSetup::solve()
    {
        ScopedTimer timer("Coupling solve");
        CouplingResult result;

        if (!isCoupled())
        {
            if (hasElectrostatics()) { electrostatics->assemble(); electrostatics->solve(); }
            return result;
        }

        for (int i = 0; i < couplingMaxIter_; ++i)
        {
            electrostatics->assemble(); electrostatics->solve();
            heatTransfer->assemble(); heatTransfer->solve();
            
            Real err = computeCouplingError();
            result.iterations = i + 1;
            result.residual = err;
            LOG_INFO << "Coupling iteration " << (i + 1) << ", residual = " << err;
            if (err < couplingTol_) { result.converged = true; break; }
        }

        if (hasStructural()) { structural->assemble(); structural->solve(); }
        return result;
    }

    Real PhysicsProblemSetup::computeCouplingError()
    {
        if (!heatTransfer) return 0.0;
        const auto& T = heatTransfer->field().values();
        if (prevT_.size() == 0) { prevT_ = T; return 1.0; }
        Real diff = (T - prevT_).norm();
        prevT_ = T;
        return diff / (T.norm() + 1e-15);
    }

} // namespace mpfem

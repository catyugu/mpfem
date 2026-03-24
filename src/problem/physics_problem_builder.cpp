#include "physics_problem_builder.hpp"
#include "physics/field_values.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include <optional>

namespace mpfem
{

    namespace
    {

        bool hasValidValue(const std::optional<Real> &value)
        {
            return value.has_value() && value.value() > 0.0;
        }

        Real parseValue(const std::map<std::string, std::string> &params,
                        const std::string &key,
                        const CaseDefinition &caseDef,
                        Real defaultVal = 0.0)
        {
            auto it = params.find(key);
            if (it == params.end())
                return defaultVal;
            std::string numStr = it->second;
            if (auto pos = numStr.find('['); pos != std::string::npos)
                numStr = numStr.substr(0, pos);
            try
            {
                return std::stod(numStr);
            }
            catch (...)
            {
                try
                {
                    return caseDef.getVariable(numStr);
                }
                catch (...)
                {
                    return defaultVal;
                }
            }
        }

        double getInitialCondition(const CaseDefinition &caseDef,
                                   const std::string &fieldKind,
                                   double defaultVal)
        {
            for (const auto &ic : caseDef.initialConditions)
            {
                if (ic.fieldKind == fieldKind)
                {
                    return ic.value;
                }
            }
            return defaultVal;
        }

    } // namespace

    std::unique_ptr<Problem> PhysicsProblemBuilder::build(const std::string &caseDir)
    {
        std::string casePath = caseDir + "/case.xml";
        LOG_INFO << "Reading case from " << casePath;

        CaseDefinition caseDef;
        CaseXmlReader::readFromFile(casePath, caseDef);

        std::unique_ptr<Problem> problem;

        // Determine problem type based on study type
        const bool isTransient = (caseDef.studyType == "transient");

        TransientProblem *transientProblem = nullptr;

        if (isTransient)
        {
            auto transientProb = std::make_unique<TransientProblem>();
            transientProblem = transientProb.get();

            // Configure time stepping parameters
            transientProb->startTime = caseDef.timeConfig.start;
            transientProb->endTime = caseDef.timeConfig.end;
            transientProb->timeStep = caseDef.timeConfig.step;

            // Parse time scheme
            if (caseDef.timeConfig.scheme == "BDF2")
            {
                transientProb->scheme = TimeScheme::BDF2;
            }
            else if (caseDef.timeConfig.scheme == "CrankNicolson")
            {
                transientProb->scheme = TimeScheme::CrankNicolson;
            }
            else
            {
                // Default to BDF1 / BackwardEuler
                transientProb->scheme = TimeScheme::BackwardEuler;
            }

            problem = std::move(transientProb);
        }
        else
        {
            problem = std::make_unique<SteadyProblem>();
        }

        if (caseDef.couplingConfig.maxIterations > 0)
        {
            problem->couplingMaxIter = caseDef.couplingConfig.maxIterations;
            problem->couplingTol = caseDef.couplingConfig.tolerance;
        }

        // Set common problem properties
        problem->caseName = caseDef.caseName;
        problem->caseDef = caseDef;

        std::string meshPath = caseDir + "/" + caseDef.meshPath;
        LOG_INFO << "Reading mesh from " << meshPath;

        problem->mesh = std::make_unique<Mesh>(MphtxtReader::read(meshPath));
        LOG_INFO << "Mesh loaded: " << problem->mesh->numVertices() << " vertices, "
                 << problem->mesh->numElements() << " elements";

        std::string matPath = caseDir + "/" + caseDef.materialsPath;
        LOG_INFO << "Reading materials from " << matPath;

        MaterialXmlReader::readFromFile(matPath, problem->materials);

        buildSolvers(*problem);

        // Initialize transient after building solvers
        if (isTransient)
        {
            // BDF2 is a 2-step method requiring T^{n+1}, T^n, T^{n-1} -> historyDepth = 3
            // BDF1 is a 1-step method requiring T^{n+1}, T^n -> historyDepth = 2
            int historyDepth = (transientProblem->scheme == TimeScheme::BDF2) ? 3 : 2;
            transientProblem->initializeTransient(historyDepth);
        }

        return problem;
    }

    void PhysicsProblemBuilder::buildSolvers(Problem &problem)
    {
        const auto &caseDef = problem.caseDef;

        // Build domain material map
        for (const auto &assign : caseDef.materialAssignments)
            for (int domId : assign.domainIds)
                problem.domainMaterial[domId] = assign.materialTag;

        // Use IPhysicsBuilder factory for explicit dispatch over caseDef.physics map
        for (const auto &[kind, physics] : caseDef.physics)
        {
            auto builder = createPhysicsBuilder(kind);
            if (builder)
                builder->build(problem, physics);
        }

        if (problem.hasJouleHeating() || problem.hasThermalExpansion())
        {
            setupCoupling(problem);
        }
    }

    void ElectrostaticsBuilder::build(Problem &problem, const CaseDefinition::Physics &physics)
    {
        LOG_INFO << "Building electrostatics solver, order = " << physics.order;
        problem.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
        problem.electrostatics->setSolverConfig(physics.solver);

        double icValue = getInitialCondition(problem.caseDef, "electrostatics", 0.0);
        problem.electrostatics->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            const auto *mat = problem.materials.getMaterial(matTag);
            if (!mat || !mat->electricConductivity.has_value())
                continue;

            std::string key = "conductivity_" + std::to_string(domId);
            problem.setCoef(key, constantMatrixCoefficient(mat->electricConductivity.value()));
            problem.electrostatics->setElectricalConductivity({domId}, problem.getCoef<MatrixCoefficient>(key));
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind != "voltage" || bc.ids.empty())
                continue;

            Real voltage = parseValue(bc.params, "value", problem.caseDef);
            std::string key = "voltage_bc_" + std::to_string(*bc.ids.begin());
            problem.setCoef(key, constantCoefficient(voltage));
            problem.electrostatics->addVoltageBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<Coefficient>(key));
        }
    }

    void HeatTransferBuilder::build(Problem &problem, const CaseDefinition::Physics &physics)
    {
        LOG_INFO << "Building heat transfer solver, order = " << physics.order;
        problem.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
        problem.heatTransfer->setSolverConfig(physics.solver);

        double icValue = getInitialCondition(problem.caseDef, "heat_transfer", 293.15);
        problem.heatTransfer->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            const auto *mat = problem.materials.getMaterial(matTag);
            if (!mat)
                continue;

            if (mat->thermalConductivity.has_value())
            {
                std::string key = "thermal_conductivity_" + std::to_string(domId);
                problem.setCoef(key, constantMatrixCoefficient(mat->thermalConductivity.value()));
                problem.heatTransfer->setThermalConductivity({domId}, problem.getCoef<MatrixCoefficient>(key));
            }

            if (hasValidValue(mat->density))
            {
                std::string key = "density_" + std::to_string(domId);
                problem.setCoef(key, constantCoefficient(mat->density.value()));
                problem.heatTransfer->setDensity({domId}, problem.getCoef<Coefficient>(key));
            }

            if (hasValidValue(mat->heatCapacity))
            {
                std::string key = "heat_capacity_" + std::to_string(domId);
                problem.setCoef(key, constantCoefficient(mat->heatCapacity.value()));
                problem.heatTransfer->setSpecificHeat({domId}, problem.getCoef<Coefficient>(key));
            }
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.ids.empty())
                continue;

            if (bc.kind == "temperature")
            {
                Real temp = parseValue(bc.params, "value", problem.caseDef);
                std::string key = "temp_bc_" + std::to_string(*bc.ids.begin());
                problem.setCoef(key, constantCoefficient(temp));
                problem.heatTransfer->addTemperatureBC({bc.ids.begin(), bc.ids.end()}, problem.getCoef<Coefficient>(key));
                continue;
            }

            if (bc.kind == "convection")
            {
                Real h = parseValue(bc.params, "h", problem.caseDef, 5.0);
                Real tinf = parseValue(bc.params, "T_inf", problem.caseDef, 293.15);
                std::string hKey = "conv_h_" + std::to_string(*bc.ids.begin());
                std::string tinfKey = "conv_tinf_" + std::to_string(*bc.ids.begin());
                problem.setCoef(hKey, constantCoefficient(h));
                problem.setCoef(tinfKey, constantCoefficient(tinf));
                problem.heatTransfer->addConvectionBC({bc.ids.begin(), bc.ids.end()},
                                                      problem.getCoef<Coefficient>(hKey),
                                                      problem.getCoef<Coefficient>(tinfKey));
            }
        }
    }

    void StructuralBuilder::build(Problem &problem, const CaseDefinition::Physics &physics)
    {
        LOG_INFO << "Building structural solver, order = " << physics.order;
        problem.structural = std::make_unique<StructuralSolver>(physics.order);
        problem.structural->setSolverConfig(physics.solver);

        double icValue = getInitialCondition(problem.caseDef, "solid_mechanics", 0.0);
        problem.structural->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            const auto *mat = problem.materials.getMaterial(matTag);
            if (!mat)
                continue;

            Real E = mat->youngModulus.value_or(0.0);
            Real nu = mat->poissonRatio.value_or(0.0);

            std::string eKey = "young_" + std::to_string(domId);
            std::string nuKey = "poisson_" + std::to_string(domId);
            problem.setCoef(eKey, constantCoefficient(E));
            problem.setCoef(nuKey, constantCoefficient(nu));
            problem.structural->setYoungModulus({domId}, problem.getCoef<Coefficient>(eKey));
            problem.structural->setPoissonRatio({domId}, problem.getCoef<Coefficient>(nuKey));
        }

        for (const auto &bc : physics.boundaries)
        {
            if (bc.kind != "fixed_constraint" || bc.ids.empty())
                continue;

            std::string key = "fixed_disp_" + std::to_string(*bc.ids.begin());
            problem.setCoef(key, constantVectorCoefficient(0.0, 0.0, 0.0));
            problem.structural->addFixedDisplacementBC({bc.ids.begin(), bc.ids.end()},
                                                       problem.getCoef<VectorCoefficient>(key));
        }
    }

    void PhysicsProblemBuilder::setupCoupling(Problem &problem)
    {
        // Storage for coupling coefficients - kept alive in problem.coefficients
        DomainMappedMatrixCoefficient tempDepSigmaMap;

        // Build temperature-dependent conductivity map (matrix form)
        for (const auto &[domId, matTag] : problem.domainMaterial)
        {
            const auto *mat = problem.materials.getMaterial(matTag);
            if (!mat)
                continue;

            if (hasValidValue(mat->rho0))
            {
                Real rho0 = mat->rho0.value();
                Real alpha = mat->alpha.value_or(0.0);
                Real tref = mat->tref.value_or(298.0);
                const GridFunction *T_field = &problem.heatTransfer->field();

                auto coef = std::make_unique<MatrixFunctionCoefficient>(
                    [T_field, rho0, alpha, tref](ElementTransform &trans, Matrix3 &result, Real)
                    {
                        Real temp = tref;
                        if (T_field)
                        {
                            const auto &ip = trans.integrationPoint();
                            temp = T_field->eval(trans.elementIndex(), &ip.xi);
                        }
                        Real factor = 1.0 + alpha * (temp - tref);
                        Real sigma = 1.0 / (rho0 * (factor > 0 ? factor : 1e-10));
                        result = Matrix3::Identity() * sigma;
                    });
                tempDepSigmaMap.set(domId, coef.get());
                std::string key = "tempDepSigma_" + std::to_string(domId);
                problem.setCoef(key, std::move(coef));
                continue;
            }

            if (auto *c = problem.getCoef<MatrixCoefficient>("conductivity_" + std::to_string(domId)))
                tempDepSigmaMap.set(domId, c);
        }

        auto tempDepSigmaMapPtr = std::make_unique<DomainMappedMatrixCoefficient>(std::move(tempDepSigmaMap));
        problem.setCoef("tempDepSigmaMap", std::move(tempDepSigmaMapPtr));
        problem.electrostatics->setElectricalConductivity(problem.getCoef<MatrixCoefficient>("tempDepSigmaMap"));

        // Setup coupled physics
        for (const auto &cp : problem.caseDef.coupledPhysicsDefinitions)
        {
            if (cp.kind == "joule_heating")
            {
                // Create Joule heat coefficient using lambda
                const GridFunction *V_field = &problem.electrostatics->field();
                const MatrixCoefficient *sigma_coef = &problem.electrostatics->electricalConductivity();

                auto jouleHeat = std::make_unique<ScalarCoefficient>(
                    [V_field, sigma_coef](ElementTransform &trans, Real &result, Real t)
                    {
                        if (!V_field || !sigma_coef)
                        {
                            result = 0.0;
                            return;
                        }
                        Matrix3 sigma_mat;
                        sigma_coef->eval(trans, sigma_mat, t);
                        Vector3 g = V_field->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
                        // For anisotropic: Q = g^T * sigma * g
                        result = g.transpose() * sigma_mat * g;
                    });

                DomainMappedScalarCoefficient jouleHeatMap;
                jouleHeatMap.set({cp.domainIds.begin(), cp.domainIds.end()}, jouleHeat.get());
                problem.setCoef("jouleHeat", std::move(jouleHeat));
                auto jouleHeatMapPtr = std::make_unique<DomainMappedScalarCoefficient>(std::move(jouleHeatMap));
                problem.setCoef("jouleHeatMap", std::move(jouleHeatMapPtr));
                problem.heatTransfer->setHeatSource(problem.getCoef<Coefficient>("jouleHeatMap"));
                LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
            }
            else if (cp.kind == "thermal_expansion")
            {
                for (int domId : cp.domainIds)
                {
                    const auto domIt = problem.domainMaterial.find(domId);
                    if (domIt == problem.domainMaterial.end())
                        continue;

                    const auto *mat = problem.materials.getMaterial(domIt->second);
                    if (!mat || !hasValidValue(mat->thermalExpansion))
                        continue;

                    Real alpha_T = mat->thermalExpansion.value();
                    Real T_ref = 293.15;
                    const GridFunction *T_field = &problem.heatTransfer->field();

                    std::string key = "thermalExp_" + std::to_string(domId);
                    auto coef = std::make_unique<ScalarCoefficient>(
                        [T_field, alpha_T, T_ref](ElementTransform &trans, Real &result, Real)
                        {
                            Real T = T_ref;
                            if (T_field)
                            {
                                const auto &ip = trans.integrationPoint();
                                T = T_field->eval(trans.elementIndex(), &ip.xi);
                            }
                            result = alpha_T * (T - T_ref);
                        });
                    problem.structural->setThermalExpansion({domId}, coef.get());
                    problem.setCoef(key, std::move(coef));
                }
                LOG_INFO << "Thermal expansion coupling enabled";
            }
        }
    }

} // namespace mpfem

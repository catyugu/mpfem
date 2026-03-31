#include "physics_problem_builder.hpp"
#include "problem.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"
#include "physics/field_values.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem/expression_coefficient_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "io/problem_input_loader.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include <cstdint>

namespace mpfem
{

    namespace
    {

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

        const std::string &requireParam(const std::map<std::string, std::string> &params,
                                        const std::string &key)
        {
            auto it = params.find(key);
            if (it == params.end())
            {
                throw ArgumentException("Missing required parameter: " + key);
            }
            return it->second;
        }

        TimeScheme parseTimeScheme(const std::string &scheme)
        {
            if (scheme == "BDF1")
            {
                return TimeScheme::BDF1;
            }
            if (scheme == "BDF2")
            {
                return TimeScheme::BDF2;
            }
            throw ArgumentException("Unsupported transient time scheme: " + scheme + ". Supported values: BDF1, BDF2.");
        }

        constexpr std::uint64_t kLocalTagSeed = 1469598103934665603ull;

        const GridFunction *resolveRuntimeField(const Problem &problem, std::string_view symbol)
        {
            if (symbol == "T")
            {
                return problem.heatTransfer ? &problem.heatTransfer->field() : nullptr;
            }
            if (symbol == "V")
            {
                return problem.electrostatics ? &problem.electrostatics->field() : nullptr;
            }
            return nullptr;
        }

        template <typename PointerRange>
        std::uint64_t combinePointerStateTags(std::uint64_t seed, const PointerRange &pointers)
        {
            std::uint64_t tag = seed;
            for (const auto *ptr : pointers)
            {
                if (ptr)
                {
                    tag = combineTag(tag, ptr->stateTag());
                }
            }
            return tag;
        }

        std::uint64_t combineFieldRevisionTag(std::uint64_t seed, const GridFunction *field)
        {
            if (!field)
            {
                return seed;
            }
            return combineTag(seed, field->revision());
        }

        RuntimeExpressionResolvers makeRuntimeExpressionResolvers(Problem &problem)
        {
            RuntimeExpressionResolvers resolvers;

            resolvers.symbolResolver =
                [&problem](std::string_view symbol,
                           ElementTransform &trans,
                           Real,
                           double &value)
            {
                const GridFunction *field = resolveRuntimeField(problem, symbol);
                if (!field)
                {
                    return false;
                }

                const auto &ip = trans.integrationPoint();
                value = field->eval(trans.elementIndex(), &ip.xi);
                return true;
            };

            resolvers.stateTagResolver =
                [&problem](std::string_view symbol) -> std::uint64_t
            {
                const GridFunction *field = resolveRuntimeField(problem, symbol);
                if (!field)
                {
                    return DynamicCoefficientTag;
                }
                return field->revision();
            };

            return resolvers;
        }

        std::unique_ptr<Coefficient> makeScalarExpressionCoefficient(Problem &problem,
                                                                     const std::string &expression)
        {
            return createRuntimeScalarExpressionCoefficient(
                expression,
                problem.caseDef,
                makeRuntimeExpressionResolvers(problem));
        }

        std::unique_ptr<MatrixCoefficient> makeMatrixExpressionCoefficient(Problem &problem,
                                                                           const std::string &expression)
        {
            return createRuntimeMatrixExpressionCoefficient(
                expression,
                problem.caseDef,
                makeRuntimeExpressionResolvers(problem));
        }

    } // namespace

    namespace PhysicsProblemBuilder
    {
        void buildSolvers(Problem &problem);
        void setupCoupling(Problem &problem);
        void buildElectrostatics(Problem &problem, const CaseDefinition::Physics &physics);
        void buildHeatTransfer(Problem &problem, const CaseDefinition::Physics &physics);
        void buildStructural(Problem &problem, const CaseDefinition::Physics &physics);

        std::unique_ptr<Problem> build(const std::string &caseDir,
                                       const ProblemInputLoader &inputLoader)
        {
            ProblemInputData input = inputLoader.load(caseDir);

            std::unique_ptr<Problem> problem;

            // Determine problem type based on study type
            const bool isTransient = (input.caseDefinition.studyType == "transient");

            TransientProblem *transientProblem = nullptr;

            if (isTransient)
            {
                auto transientProb = std::make_unique<TransientProblem>();
                transientProblem = transientProb.get();

                // Configure time stepping parameters
                transientProb->startTime = input.caseDefinition.timeConfig.start;
                transientProb->endTime = input.caseDefinition.timeConfig.end;
                transientProb->timeStep = input.caseDefinition.timeConfig.step;

                transientProb->scheme = parseTimeScheme(input.caseDefinition.timeConfig.scheme);

                problem = std::move(transientProb);
            }
            else
            {
                problem = std::make_unique<SteadyProblem>();
            }

            if (input.caseDefinition.couplingConfig.maxIterations > 0)
            {
                problem->couplingMaxIter = input.caseDefinition.couplingConfig.maxIterations;
                problem->couplingTol = input.caseDefinition.couplingConfig.tolerance;
            }

            problem->caseName = input.caseDefinition.caseName;
            problem->caseDef = std::move(input.caseDefinition);
            problem->mesh = std::move(input.mesh);
            problem->materials = std::move(input.materials);

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

        std::unique_ptr<Problem> build(const std::string &caseDir)
        {
            std::unique_ptr<ProblemInputLoader> inputLoader = createXmlProblemInputLoader();
            return build(caseDir, *inputLoader);
        }

        void buildSolvers(Problem &problem)
        {
            const auto &caseDef = problem.caseDef;

            // Build domain material map
            for (const auto &assign : caseDef.materialAssignments)
                for (int domId : assign.domainIds)
                    problem.domainMaterial[domId] = assign.materialTag;

            for (const auto &[kind, physics] : caseDef.physics)
            {
                if (kind == "electrostatics")
                {
                    buildElectrostatics(problem, physics);
                    continue;
                }
                if (kind == "heat_transfer")
                {
                    buildHeatTransfer(problem, physics);
                    continue;
                }
                if (kind == "solid_mechanics")
                {
                    buildStructural(problem, physics);
                }
            }

            if (problem.hasJouleHeating() || problem.hasThermalExpansion())
            {
                setupCoupling(problem);
            }
        }

        void buildElectrostatics(Problem &problem, const CaseDefinition::Physics &physics)
        {
            LOG_INFO << "Building electrostatics solver, order = " << physics.order;
            problem.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
            problem.electrostatics->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, "electrostatics", 0.0);
            problem.electrostatics->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (const auto &assign : problem.caseDef.materialAssignments)
            {
                const auto *mat = problem.materials.getMaterial(assign.materialTag);
                if (!mat)
                    continue;

                if (!mat->hasMatrix("electricconductivity"))
                    continue;

                const int keyDomain = assign.domainIds.empty() ? -1 : *assign.domainIds.begin();
                std::string key = "conductivity_" + assign.materialTag + "_" + std::to_string(keyDomain);
                problem.setMatrixCoef(
                    key,
                    makeMatrixExpressionCoefficient(problem, mat->matrixExpression("electricconductivity")));

                const MatrixCoefficient *sigma = problem.getMatrixCoef(key);
                problem.electrostatics->setElectricalConductivity(assign.domainIds, sigma);
                for (int domId : assign.domainIds)
                {
                    problem.conductivityByDomain[domId] = sigma;
                }
            }

            for (const auto &bc : physics.boundaries)
            {
                if (bc.kind != "voltage" || bc.ids.empty())
                    continue;

                std::string key = "voltage_bc_" + std::to_string(*bc.ids.begin());
                problem.setScalarCoef(key, makeScalarExpressionCoefficient(problem, requireParam(bc.params, "value")));
                problem.electrostatics->addVoltageBC({bc.ids.begin(), bc.ids.end()}, problem.getScalarCoef(key));
            }
        }

        void buildHeatTransfer(Problem &problem, const CaseDefinition::Physics &physics)
        {
            LOG_INFO << "Building heat transfer solver, order = " << physics.order;
            problem.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
            problem.heatTransfer->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, "heat_transfer", 293.15);
            problem.heatTransfer->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (const auto &assign : problem.caseDef.materialAssignments)
            {
                const auto *mat = problem.materials.getMaterial(assign.materialTag);
                if (!mat)
                    continue;

                const int keyDomain = assign.domainIds.empty() ? -1 : *assign.domainIds.begin();

                if (mat->hasMatrix("thermalconductivity"))
                {
                    std::string key = "thermal_conductivity_" + assign.materialTag + "_" + std::to_string(keyDomain);
                    problem.setMatrixCoef(
                        key,
                        makeMatrixExpressionCoefficient(problem, mat->matrixExpression("thermalconductivity")));
                    problem.heatTransfer->setThermalConductivity(assign.domainIds, problem.getMatrixCoef(key));
                }

                if (mat->hasScalar("density") && mat->hasScalar("heatcapacity"))
                {
                    std::string rhoKey = "density_" + assign.materialTag + "_" + std::to_string(keyDomain);
                    std::string cpKey = "heat_capacity_" + assign.materialTag + "_" + std::to_string(keyDomain);
                    problem.setScalarCoef(rhoKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("density")));
                    problem.setScalarCoef(cpKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("heatcapacity")));
                    problem.heatTransfer->setMassProperties(assign.domainIds,
                                                            problem.getScalarCoef(rhoKey),
                                                            problem.getScalarCoef(cpKey));
                }
            }

            for (const auto &bc : physics.boundaries)
            {
                if (bc.ids.empty())
                    continue;

                if (bc.kind == "temperature")
                {
                    std::string key = "temp_bc_" + std::to_string(*bc.ids.begin());
                    problem.setScalarCoef(key, makeScalarExpressionCoefficient(problem, requireParam(bc.params, "value")));
                    problem.heatTransfer->addTemperatureBC({bc.ids.begin(), bc.ids.end()}, problem.getScalarCoef(key));
                    continue;
                }

                if (bc.kind == "convection")
                {
                    std::string hKey = "conv_h_" + std::to_string(*bc.ids.begin());
                    std::string tinfKey = "conv_tinf_" + std::to_string(*bc.ids.begin());
                    problem.setScalarCoef(hKey, makeScalarExpressionCoefficient(problem, requireParam(bc.params, "h")));
                    problem.setScalarCoef(tinfKey, makeScalarExpressionCoefficient(problem, requireParam(bc.params, "T_inf")));
                    problem.heatTransfer->addConvectionBC({bc.ids.begin(), bc.ids.end()},
                                                          problem.getScalarCoef(hKey),
                                                          problem.getScalarCoef(tinfKey));
                }
            }
        }

        void buildStructural(Problem &problem, const CaseDefinition::Physics &physics)
        {
            LOG_INFO << "Building structural solver, order = " << physics.order;
            problem.structural = std::make_unique<StructuralSolver>(physics.order);
            problem.structural->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, "solid_mechanics", 0.0);
            problem.structural->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (const auto &assign : problem.caseDef.materialAssignments)
            {
                const auto *mat = problem.materials.getMaterial(assign.materialTag);
                if (!mat)
                    continue;

                if (!mat->hasScalar("E") || !mat->hasScalar("nu"))
                {
                    throw ArgumentException("Material missing required structural properties E/nu: " + assign.materialTag);
                }

                const int keyDomain = assign.domainIds.empty() ? -1 : *assign.domainIds.begin();
                std::string eKey = "young_" + assign.materialTag + "_" + std::to_string(keyDomain);
                std::string nuKey = "poisson_" + assign.materialTag + "_" + std::to_string(keyDomain);
                problem.setScalarCoef(eKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("E")));
                problem.setScalarCoef(nuKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("nu")));
                const Coefficient *E = problem.getScalarCoef(eKey);
                const Coefficient *nu = problem.getScalarCoef(nuKey);
                problem.structural->addElasticity(assign.domainIds, E, nu);
                for (int domId : assign.domainIds)
                {
                    problem.youngModulusByDomain[domId] = E;
                    problem.poissonRatioByDomain[domId] = nu;
                }
            }

            for (const auto &bc : physics.boundaries)
            {
                if (bc.kind != "fixed_constraint" || bc.ids.empty())
                    continue;

                std::string key = "fixed_disp_" + std::to_string(*bc.ids.begin());
                problem.setVectorCoef(key, constantVectorCoefficient(0.0, 0.0, 0.0));
                problem.structural->addFixedDisplacementBC({bc.ids.begin(), bc.ids.end()},
                                                           problem.getVectorCoef(key));
            }
        }

        void setupJouleHeating(Problem &problem, const CoupledPhysicsDefinition &cp)
        {
            const GridFunction *V_field = &problem.electrostatics->field();
            const int maxDomainId = cp.domainIds.empty() ? 0 : *cp.domainIds.rbegin();
            
            std::vector<const MatrixCoefficient *> sigmaByDomain(static_cast<size_t>(maxDomainId + 1), nullptr);
            std::set<int> activeDomains;
            
            for (int domId : cp.domainIds) {
                auto sigmaIt = problem.conductivityByDomain.find(domId);
                if (sigmaIt == problem.conductivityByDomain.end()) {
                    LOG_WARN << "No conductivity for domain " << domId << " in joule heating coupling, skipping";
                    continue;
                }
                sigmaByDomain[static_cast<size_t>(domId)] = sigmaIt->second;
                activeDomains.insert(domId);
            }

            if (activeDomains.empty()) {
                LOG_WARN << "No valid domains for joule heating coupling";
                return;
            }

            auto jouleHeat = std::make_unique<FunctionCoefficient>(
                [V_field, sigmaByDomain](ElementTransform &trans, Real &result, Real t) {
                    const int domId = static_cast<int>(trans.attribute());
                    if (domId < 0 || static_cast<size_t>(domId) >= sigmaByDomain.size()) {
                        result = 0.0;
                        return;
                    }
                    const MatrixCoefficient *sigmaCoef = sigmaByDomain[static_cast<size_t>(domId)];
                    if (!sigmaCoef) {
                        result = 0.0;
                        return;
                    }
                    Matrix3 sigma_mat;
                    sigmaCoef->eval(trans, sigma_mat, t);
                    Vector3 g = V_field->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
                    result = g.transpose() * sigma_mat * g;
                },
                [&problem, sigmaByDomain]() -> std::uint64_t {
                    std::uint64_t tag = combineFieldRevisionTag(kLocalTagSeed,
                        problem.electrostatics ? &problem.electrostatics->field() : nullptr);
                    return combinePointerStateTags(tag, sigmaByDomain);
                });

            std::string jouleKey = "jouleHeat_" + cp.name;
            problem.setScalarCoef(jouleKey, std::move(jouleHeat));
            problem.heatTransfer->setHeatSource(activeDomains, problem.getScalarCoef(jouleKey));
            LOG_INFO << "Joule heating domains: " << activeDomains.size() << " domains";
        }

        void setupThermalExpansion(Problem &problem, const CoupledPhysicsDefinition &cp)
        {
            auto physicsIt = problem.caseDef.physics.find("solid_mechanics");
            MPFEM_ASSERT(physicsIt != problem.caseDef.physics.end(),
                         "solid_mechanics physics block not found for thermal expansion coupling");
            Real T_ref = physicsIt->second.referenceTemperature;

            const int maxDomainId = cp.domainIds.empty() ? 0 : *cp.domainIds.rbegin();
            std::vector<const MatrixCoefficient *> alphaByDomain(static_cast<size_t>(maxDomainId + 1), nullptr);
            std::vector<const Coefficient *> youngByDomain(static_cast<size_t>(maxDomainId + 1), nullptr);
            std::vector<const Coefficient *> nuByDomain(static_cast<size_t>(maxDomainId + 1), nullptr);
            std::set<int> activeDomains;
            
            for (int domId : cp.domainIds) {
                const auto domIt = problem.domainMaterial.find(domId);
                if (domIt == problem.domainMaterial.end())
                    continue;

                const auto *mat = problem.materials.getMaterial(domIt->second);
                if (!mat || !mat->hasMatrix("thermalexpansioncoefficient")) {
                    LOG_WARN << "Material lacks thermal expansion coefficient for domain " << domId;
                    continue;
                }

                std::string alphaKey = "alphaThermal_" + std::to_string(domId);
                problem.setMatrixCoef(alphaKey,
                    makeMatrixExpressionCoefficient(problem, mat->matrixExpression("thermalexpansioncoefficient")));

                const MatrixCoefficient *alphaCoef = problem.getMatrixCoef(alphaKey);
                MPFEM_ASSERT(alphaCoef != nullptr, "Failed to build thermal expansion coefficient.");

                auto eIt = problem.youngModulusByDomain.find(domId);
                auto nuIt = problem.poissonRatioByDomain.find(domId);
                if (eIt == problem.youngModulusByDomain.end() || nuIt == problem.poissonRatioByDomain.end()) {
                    LOG_WARN << "Missing structural material coefficients for domain " << domId;
                    continue;
                }

                alphaByDomain[static_cast<size_t>(domId)] = alphaCoef;
                youngByDomain[static_cast<size_t>(domId)] = eIt->second;
                nuByDomain[static_cast<size_t>(domId)] = nuIt->second;
                activeDomains.insert(domId);
            }

            if (activeDomains.empty()) {
                LOG_WARN << "No valid domains for thermal expansion coupling";
                return;
            }

            std::string key = "thermalStress_" + cp.name;
            auto coef = std::make_unique<MatrixFunctionCoefficient>(
                [&problem, alphaByDomain, youngByDomain, nuByDomain, T_ref](ElementTransform &trans, Matrix3 &result, Real t) {
                    const int domId = static_cast<int>(trans.attribute());
                    if (domId < 0 || static_cast<size_t>(domId) >= alphaByDomain.size()) {
                        result.setZero();
                        return;
                    }
                    const MatrixCoefficient *alphaCoef = alphaByDomain[static_cast<size_t>(domId)];
                    const Coefficient *eCoef = youngByDomain[static_cast<size_t>(domId)];
                    const Coefficient *nuCoef = nuByDomain[static_cast<size_t>(domId)];
                    if (!alphaCoef || !eCoef || !nuCoef) {
                        result.setZero();
                        return;
                    }

                    Matrix3 alpha;
                    alphaCoef->eval(trans, alpha, t);

                    Real T = T_ref;
                    if (problem.heatTransfer) {
                        const auto &ip = trans.integrationPoint();
                        T = problem.heatTransfer->field().eval(trans.elementIndex(), &ip.xi);
                    }

                    Real E_val = 0.0, nu_val = 0.0;
                    eCoef->eval(trans, E_val, t);
                    nuCoef->eval(trans, nu_val, t);

                    const Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
                    const Real mu = E_val / (2.0 * (1.0 + nu_val));

                    const Matrix3 eps = alpha * (T - T_ref);
                    const Matrix3 epsSym = 0.5 * (eps + eps.transpose());

                    result = 2.0 * mu * epsSym;
                    result.diagonal().array() += lambda * epsSym.trace();
                },
                [&problem, alphaByDomain, youngByDomain, nuByDomain]() -> std::uint64_t {
                    std::uint64_t tag = combineFieldRevisionTag(kLocalTagSeed,
                        problem.heatTransfer ? &problem.heatTransfer->field() : nullptr);
                    tag = combinePointerStateTags(tag, alphaByDomain);
                    tag = combinePointerStateTags(tag, youngByDomain);
                    return combinePointerStateTags(tag, nuByDomain);
                });
            
            problem.structural->setStrainLoad(activeDomains, coef.get());
            problem.setMatrixCoef(key, std::move(coef));
            LOG_INFO << "Thermal expansion coupling enabled (T_ref = " << T_ref << " K)";
        }

        void setupCoupling(Problem &problem)
        {
            for (const auto &cp : problem.caseDef.coupledPhysicsDefinitions) {
                if (cp.kind == "joule_heating") {
                    setupJouleHeating(problem, cp);
                } else if (cp.kind == "thermal_expansion") {
                    setupThermalExpansion(problem, cp);
                }
            }
        }

    } // namespace PhysicsProblemBuilder

} // namespace mpfem

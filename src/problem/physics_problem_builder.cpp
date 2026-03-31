#include "physics_problem_builder.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "io/problem_input_loader.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/field_values.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "problem/expression_coefficient_factory.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"
#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace mpfem {

    namespace {

        constexpr std::string_view kPhysicsElectrostatics = "electrostatics";
        constexpr std::string_view kPhysicsHeatTransfer = "heat_transfer";
        constexpr std::string_view kPhysicsSolidMechanics = "solid_mechanics";

        constexpr std::string_view kPropElectricConductivity = "electricconductivity";
        constexpr std::string_view kPropThermalConductivity = "thermalconductivity";
        constexpr std::string_view kPropDensity = "density";
        constexpr std::string_view kPropHeatCapacity = "heatcapacity";
        constexpr std::string_view kPropYoungModulus = "E";
        constexpr std::string_view kPropPoissonRatio = "nu";
        constexpr std::string_view kPropThermalExpansion = "thermalexpansioncoefficient";
        double getInitialCondition(const CaseDefinition& caseDef,
            std::string_view fieldKind,
            double defaultVal)
        {
            for (const auto& ic : caseDef.initialConditions) {
                if (ic.fieldKind == fieldKind) {
                    return ic.value;
                }
            }
            return defaultVal;
        }

        TimeScheme parseTimeScheme(const std::string& scheme)
        {
            if (scheme == "BDF1") {
                return TimeScheme::BDF1;
            }
            if (scheme == "BDF2") {
                return TimeScheme::BDF2;
            }
            throw ArgumentException("Unsupported transient time scheme: " + scheme + ". Supported values: BDF1, BDF2.");
        }

        std::unique_ptr<Coefficient> makeScalarExpressionCoefficient(Problem& problem,
            const std::string& expression);

        std::unique_ptr<MatrixCoefficient> makeMatrixExpressionCoefficient(Problem& problem,
            const std::string& expression);

        template <typename CoefT, typename GetterFn>
        std::unordered_map<int, const CoefT*> collectDomainCoefficients(const std::set<int>& domainIds,
            GetterFn getter)
        {
            std::unordered_map<int, const CoefT*> byDomain;
            byDomain.reserve(domainIds.size());
            for (int domainId : domainIds) {
                byDomain.emplace(domainId, getter(domainId));
            }
            return byDomain;
        }

        const MatrixCoefficient* requireDomainMatrixCoefficient(Problem& problem,
            int domainId,
            std::string_view property)
        {
            if (const MatrixCoefficient* existing = problem.findDomainMatrixCoef(property, domainId)) {
                return existing;
            }

            const std::string& expression = problem.materials.matrixExpressionByDomain(domainId, property);
            return problem.setDomainMatrixCoef(std::string(property),
                domainId,
                makeMatrixExpressionCoefficient(problem, expression));
        }

        const Coefficient* requireDomainScalarCoefficient(Problem& problem,
            int domainId,
            std::string_view property)
        {
            if (const Coefficient* existing = problem.findDomainScalarCoef(property, domainId)) {
                return existing;
            }

            const std::string& expression = problem.materials.scalarExpressionByDomain(domainId, property);
            return problem.setDomainScalarCoef(std::string(property),
                domainId,
                makeScalarExpressionCoefficient(problem, expression));
        }

        constexpr std::uint64_t kLocalTagSeed = 1469598103934665603ull;

        struct RuntimeSymbolBinding {
            const GridFunction* field = nullptr;
        };

        RuntimeSymbolBinding resolveRuntimeSymbolBinding(const Problem& problem, std::string_view symbol)
        {
            if (symbol == "T") {
                return RuntimeSymbolBinding {problem.heatTransfer ? &problem.heatTransfer->field() : nullptr};
            }
            if (symbol == "V") {
                return RuntimeSymbolBinding {problem.electrostatics ? &problem.electrostatics->field() : nullptr};
            }
            return RuntimeSymbolBinding {};
        }

        template <typename PointerMap>
        std::uint64_t combinePointerMapStateTags(std::uint64_t seed, const PointerMap& pointerMap)
        {
            std::uint64_t tag = seed;
            for (const auto& [_, ptr] : pointerMap) {
                if (ptr) {
                    tag = combineTag(tag, ptr->stateTag());
                }
            }
            return tag;
        }

        std::uint64_t combineFieldRevisionTag(std::uint64_t seed, const GridFunction* field)
        {
            if (!field) {
                return seed;
            }
            return combineTag(seed, field->revision());
        }

        RuntimeExpressionResolvers makeRuntimeExpressionResolvers(Problem& problem)
        {
            RuntimeExpressionResolvers resolvers;

            resolvers.symbolResolver =
                [&problem](std::string_view symbol,
                    ElementTransform& trans,
                    Real,
                    double& value) {
                    const RuntimeSymbolBinding binding = resolveRuntimeSymbolBinding(problem, symbol);
                    if (!binding.field) {
                        return false;
                    }

                    const auto& ip = trans.integrationPoint();
                    value = binding.field->eval(trans.elementIndex(), &ip.xi);
                    return true;
                };

            resolvers.stateTagResolver =
                [&problem](std::string_view symbol) -> std::uint64_t {
                const RuntimeSymbolBinding binding = resolveRuntimeSymbolBinding(problem, symbol);
                if (!binding.field) {
                    return DynamicCoefficientTag;
                }
                return binding.field->revision();
            };

            return resolvers;
        }

        std::unique_ptr<Coefficient> makeScalarExpressionCoefficient(Problem& problem,
            const std::string& expression)
        {
            return createRuntimeScalarExpressionCoefficient(
                expression,
                problem.caseDef,
                makeRuntimeExpressionResolvers(problem));
        }

        std::unique_ptr<MatrixCoefficient> makeMatrixExpressionCoefficient(Problem& problem,
            const std::string& expression)
        {
            return createRuntimeMatrixExpressionCoefficient(
                expression,
                problem.caseDef,
                makeRuntimeExpressionResolvers(problem));
        }

    } // namespace

    namespace PhysicsProblemBuilder {
        void buildSolvers(Problem& problem);
        void setupCoupling(Problem& problem);
        void buildElectrostatics(Problem& problem, const CaseDefinition::Physics& physics);
        void buildHeatTransfer(Problem& problem, const CaseDefinition::Physics& physics);
        void buildStructural(Problem& problem, const CaseDefinition::Physics& physics);

        std::unique_ptr<Problem> build(const std::string& caseDir,
            const ProblemInputLoader& inputLoader)
        {
            ProblemInputData input = inputLoader.load(caseDir);

            std::unique_ptr<Problem> problem;

            // Determine problem type based on study type
            const bool isTransient = (input.caseDefinition.studyType == "transient");

            TransientProblem* transientProblem = nullptr;

            if (isTransient) {
                auto transientProb = std::make_unique<TransientProblem>();
                transientProblem = transientProb.get();

                // Configure time stepping parameters
                transientProb->startTime = input.caseDefinition.timeConfig.start;
                transientProb->endTime = input.caseDefinition.timeConfig.end;
                transientProb->timeStep = input.caseDefinition.timeConfig.step;

                transientProb->scheme = parseTimeScheme(input.caseDefinition.timeConfig.scheme);

                problem = std::move(transientProb);
            }
            else {
                problem = std::make_unique<SteadyProblem>();
            }

            if (input.caseDefinition.couplingConfig.maxIterations > 0) {
                problem->couplingMaxIter = input.caseDefinition.couplingConfig.maxIterations;
                problem->couplingTol = input.caseDefinition.couplingConfig.tolerance;
            }

            problem->caseName = input.caseDefinition.caseName;
            problem->caseDef = std::move(input.caseDefinition);
            problem->mesh = std::move(input.mesh);
            problem->materials = std::move(input.materials);
            problem->materials.buildDomainIndex(problem->caseDef.materialAssignments);

            buildSolvers(*problem);

            // Initialize transient after building solvers
            if (isTransient) {
                // BDF2 is a 2-step method requiring T^{n+1}, T^n, T^{n-1} -> historyDepth = 3
                // BDF1 is a 1-step method requiring T^{n+1}, T^n -> historyDepth = 2
                int historyDepth = (transientProblem->scheme == TimeScheme::BDF2) ? 3 : 2;
                transientProblem->initializeTransient(historyDepth);
            }

            return problem;
        }

        std::unique_ptr<Problem> build(const std::string& caseDir)
        {
            std::unique_ptr<ProblemInputLoader> inputLoader = createXmlProblemInputLoader();
            return build(caseDir, *inputLoader);
        }

        void buildSolvers(Problem& problem)
        {
            const auto& caseDef = problem.caseDef;

            for (const auto& [kind, physics] : caseDef.physics) {
                if (kind == "electrostatics") {
                    buildElectrostatics(problem, physics);
                    continue;
                }
                if (kind == "heat_transfer") {
                    buildHeatTransfer(problem, physics);
                    continue;
                }
                if (kind == "solid_mechanics") {
                    buildStructural(problem, physics);
                }
            }

            if (problem.hasJouleHeating() || problem.hasThermalExpansion()) {
                setupCoupling(problem);
            }
        }

        void buildElectrostatics(Problem& problem, const CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building electrostatics solver, order = " << physics.order;
            problem.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
            problem.electrostatics->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, kPhysicsElectrostatics, 0.0);
            problem.electrostatics->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (int domainId : problem.materials.domainIds()) {
                const MatrixCoefficient* sigma = requireDomainMatrixCoefficient(
                    problem,
                    domainId,
                    kPropElectricConductivity);
                problem.electrostatics->setElectricalConductivity({domainId}, sigma);
            }

            for (const auto& bc : physics.boundaries) {
                if (const auto* voltageBc = std::get_if<VoltageBoundaryCondition>(&bc)) {
                    const Coefficient* voltage = problem.ownScalarCoef(
                        makeScalarExpressionCoefficient(problem, voltageBc->value));
                    problem.electrostatics->addVoltageBC(voltageBc->ids, voltage);
                }
            }
        }

        void buildHeatTransfer(Problem& problem, const CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building heat transfer solver, order = " << physics.order;
            problem.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
            problem.heatTransfer->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, kPhysicsHeatTransfer, 293.15);
            problem.heatTransfer->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (int domainId : problem.materials.domainIds()) {
                const MatrixCoefficient* k = requireDomainMatrixCoefficient(
                    problem,
                    domainId,
                    kPropThermalConductivity);
                problem.heatTransfer->setThermalConductivity({domainId}, k);

                const Coefficient* rho = requireDomainScalarCoefficient(problem, domainId, kPropDensity);
                const Coefficient* cp = requireDomainScalarCoefficient(problem, domainId, kPropHeatCapacity);
                problem.heatTransfer->setMassProperties({domainId}, rho, cp);
            }

            for (const auto& bc : physics.boundaries) {
                if (const auto* temperatureBc = std::get_if<TemperatureBoundaryCondition>(&bc)) {
                    const Coefficient* temperature = problem.ownScalarCoef(
                        makeScalarExpressionCoefficient(problem, temperatureBc->value));
                    problem.heatTransfer->addTemperatureBC(temperatureBc->ids, temperature);
                    continue;
                }

                if (const auto* convectionBc = std::get_if<ConvectionBoundaryCondition>(&bc)) {
                    const Coefficient* h = problem.ownScalarCoef(
                        makeScalarExpressionCoefficient(problem, convectionBc->h));
                    const Coefficient* tinf = problem.ownScalarCoef(
                        makeScalarExpressionCoefficient(problem, convectionBc->tInf));
                    problem.heatTransfer->addConvectionBC(convectionBc->ids,
                        h,
                        tinf);
                }
            }
        }

        void buildStructural(Problem& problem, const CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building structural solver, order = " << physics.order;
            problem.structural = std::make_unique<StructuralSolver>(physics.order);
            problem.structural->setSolverConfig(physics.solver);

            double icValue = getInitialCondition(problem.caseDef, kPhysicsSolidMechanics, 0.0);
            problem.structural->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            for (int domainId : problem.materials.domainIds()) {
                const Coefficient* E = requireDomainScalarCoefficient(problem, domainId, kPropYoungModulus);
                const Coefficient* nu = requireDomainScalarCoefficient(problem, domainId, kPropPoissonRatio);
                problem.structural->addElasticity({domainId}, E, nu);
            }

            for (const auto& bc : physics.boundaries) {
                if (const auto* fixedBc = std::get_if<FixedConstraintBoundaryCondition>(&bc)) {
                    const VectorCoefficient* disp = problem.ownVectorCoef(
                        constantVectorCoefficient(0.0, 0.0, 0.0));
                    problem.structural->addFixedDisplacementBC(fixedBc->ids,
                        disp);
                }
            }
        }

        void setupJouleHeating(Problem& problem, const CoupledPhysicsDefinition& cp)
        {
            const GridFunction* V_field = &problem.electrostatics->field();
            const std::set<int> activeDomains = cp.domainIds;
            const auto sigmaByDomain = collectDomainCoefficients<MatrixCoefficient>(
                activeDomains,
                [&](int domainId) {
                    return requireDomainMatrixCoefficient(problem, domainId, kPropElectricConductivity);
                });

            auto jouleHeat = std::make_unique<FunctionCoefficient>(
                [V_field, sigmaByDomain](ElementTransform& trans, Real& result, Real t) {
                    const int domId = static_cast<int>(trans.attribute());
                    const MatrixCoefficient* sigmaCoef = sigmaByDomain.at(domId);
                    Matrix3 sigma_mat;
                    sigmaCoef->eval(trans, sigma_mat, t);
                    Vector3 g = V_field->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
                    result = g.transpose() * sigma_mat * g;
                },
                [&problem, sigmaByDomain]() -> std::uint64_t {
                    std::uint64_t tag = combineFieldRevisionTag(kLocalTagSeed,
                        problem.electrostatics ? &problem.electrostatics->field() : nullptr);
                    return combinePointerMapStateTags(tag, sigmaByDomain);
                });

            const Coefficient* joule = problem.ownScalarCoef(std::move(jouleHeat));
            problem.heatTransfer->setHeatSource(activeDomains, joule);
            LOG_INFO << "Joule heating domains: " << activeDomains.size() << " domains";
        }

        void setupThermalExpansion(Problem& problem, const CoupledPhysicsDefinition& cp)
        {
            auto physicsIt = problem.caseDef.physics.find(std::string(kPhysicsSolidMechanics));
            MPFEM_ASSERT(physicsIt != problem.caseDef.physics.end(),
                "solid_mechanics physics block not found for thermal expansion coupling");
            Real T_ref = physicsIt->second.referenceTemperature;

            const std::set<int> activeDomains = cp.domainIds;
            const auto alphaByDomain = collectDomainCoefficients<MatrixCoefficient>(
                activeDomains,
                [&](int domainId) {
                    return requireDomainMatrixCoefficient(problem, domainId, kPropThermalExpansion);
                });
            const auto youngByDomain = collectDomainCoefficients<Coefficient>(
                activeDomains,
                [&](int domainId) {
                    return requireDomainScalarCoefficient(problem, domainId, kPropYoungModulus);
                });
            const auto nuByDomain = collectDomainCoefficients<Coefficient>(
                activeDomains,
                [&](int domainId) {
                    return requireDomainScalarCoefficient(problem, domainId, kPropPoissonRatio);
                });

            auto coef = std::make_unique<MatrixFunctionCoefficient>(
                [&problem, alphaByDomain, youngByDomain, nuByDomain, T_ref](ElementTransform& trans, Matrix3& result, Real t) {
                    const int domId = static_cast<int>(trans.attribute());
                    const MatrixCoefficient* alphaCoef = alphaByDomain.at(domId);
                    const Coefficient* eCoef = youngByDomain.at(domId);
                    const Coefficient* nuCoef = nuByDomain.at(domId);
                    Matrix3 alpha;
                    alphaCoef->eval(trans, alpha, t);
                    Real T = T_ref;
                    const auto& ip = trans.integrationPoint();
                    T = problem.heatTransfer->field().eval(trans.elementIndex(), &ip.xi);

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
                    tag = combinePointerMapStateTags(tag, alphaByDomain);
                    tag = combinePointerMapStateTags(tag, youngByDomain);
                    return combinePointerMapStateTags(tag, nuByDomain);
                });

            const MatrixCoefficient* stressCoef = problem.ownMatrixCoef(std::move(coef));
            problem.structural->setStrainLoad(activeDomains, stressCoef);
            LOG_INFO << "Thermal expansion coupling enabled (T_ref = " << T_ref << " K)";
        }

        void setupCoupling(Problem& problem)
        {
            for (const auto& cp : problem.caseDef.coupledPhysicsDefinitions) {
                if (cp.kind == "joule_heating") {
                    setupJouleHeating(problem, cp);
                }
                else if (cp.kind == "thermal_expansion") {
                    setupThermalExpansion(problem, cp);
                }
            }
        }

    } // namespace PhysicsProblemBuilder

} // namespace mpfem

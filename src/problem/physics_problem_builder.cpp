#include "physics_problem_builder.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "expr/variable_graph.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "io/problem_input_loader.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/field_values.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"
#include <array>
#include <atomic>
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

        const VariableNode* makeScalarExpressionNode(Problem& problem,
            const std::string& expression);

        template <typename GetterFn>
        std::unordered_map<int, const VariableNode*> collectDomainNodes(const std::set<int>& domainIds,
            GetterFn getter)
        {
            std::unordered_map<int, const VariableNode*> byDomain;
            byDomain.reserve(domainIds.size());
            for (int domainId : domainIds) {
                byDomain.emplace(domainId, getter(domainId));
            }
            return byDomain;
        }

        const VariableNode* requireDomainMatrixNode(Problem& problem,
            int domainId,
            std::string_view property)
        {
            std::string name = std::string(property) + "_" + std::to_string(domainId);

            if (const VariableNode* existing = problem.globalVariables_.get(name)) {
                return existing;
            }

            const std::string& expression = problem.materials.matrixExpressionByDomain(domainId, property);
            problem.globalVariables_.registerExpression(name, expression);

            return problem.globalVariables_.get(name);
        }

        const VariableNode* requireDomainScalarNode(Problem& problem,
            int domainId,
            std::string_view property)
        {
            std::string name = std::string(property) + "_" + std::to_string(domainId);

            if (const VariableNode* existing = problem.globalVariables_.get(name)) {
                return existing;
            }

            const std::string& expression = problem.materials.scalarExpressionByDomain(domainId, property);
            problem.globalVariables_.registerExpression(name, expression);

            return problem.globalVariables_.get(name);
        }

        const VariableNode* makeScalarExpressionNode(Problem& problem,
            const std::string& expression)
        {
            static std::atomic<std::uint64_t> id {0};
            std::string name = "$expr_scalar_" + std::to_string(id++);
            problem.globalVariables_.registerExpression(name, expression);
            return problem.globalVariables_.get(name);
        }

        class JouleHeatNode final : public VariableNode {
        public:
            JouleHeatNode(const GridFunction* voltageField,
                std::unordered_map<int, const VariableNode*> sigmaByDomain)
                : voltageField_(voltageField), sigmaByDomain_(std::move(sigmaByDomain))
            {
                if (!voltageField_) {
                    MPFEM_THROW(ArgumentException, "JouleHeatNode requires voltage field.");
                }
            }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "JouleHeatNode requires ElementTransform in EvaluationContext.");
                }
                if (dest.size() != ctx.physicalPoints.size()) {
                    MPFEM_THROW(ArgumentException, "JouleHeatNode destination size mismatch.");
                }

                const int domId = static_cast<int>(ctx.transform->attribute());
                const auto it = sigmaByDomain_.find(domId);
                if (it == sigmaByDomain_.end() || !it->second) {
                    MPFEM_THROW(ArgumentException, "Missing conductivity node for domain.");
                }
                const VariableNode* sigmaNode = it->second;

                for (size_t i = 0; i < dest.size(); ++i) {
                    Vector3 refPoint = Vector3::Zero();
                    Vector3 physPoint = Vector3::Zero();
                    if (i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                        refPoint = ctx.referencePoints[i];
                    }
                    else {
                        const auto& ip = ctx.transform->integrationPoint();
                        refPoint = Vector3(ip.xi, ip.eta, ip.zeta);
                    }
                    if (i < ctx.physicalPoints.size()) {
                        physPoint = ctx.physicalPoints[i];
                    }
                    else {
                        const auto& ip = ctx.transform->integrationPoint();
                        ctx.transform->transform(ip, physPoint);
                    }

                    std::array<Vector3, 1> refPts {refPoint};
                    std::array<Vector3, 1> physPts {physPoint};
                    EvaluationContext one;
                    one.time = ctx.time;
                    one.domainId = domId;
                    one.elementId = ctx.elementId;
                    one.referencePoints = std::span<const Vector3>(refPts.data(), refPts.size());
                    one.physicalPoints = std::span<const Vector3>(physPts.data(), physPts.size());
                    one.transform = ctx.transform;

                    std::array<double, 9> sigmaValues {};
                    sigmaNode->evaluateBatch(one, std::span<double>(sigmaValues.data(), sigmaValues.size()));
                    Matrix3 sigma = Matrix3::Zero();
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            sigma(r, c) = static_cast<Real>(sigmaValues[static_cast<size_t>(r * 3 + c)]);
                        }
                    }

                    const auto& ip = ctx.transform->integrationPoint();
                    Vector3 g = voltageField_->gradient(ctx.transform->elementIndex(), &ip.xi, *ctx.transform);
                    dest[i] = g.transpose() * sigma * g;
                }
            }

        private:
            const GridFunction* voltageField_ = nullptr;
            std::unordered_map<int, const VariableNode*> sigmaByDomain_;
        };

        class ThermalExpansionStressNode final : public VariableNode {
        public:
            ThermalExpansionStressNode(const HeatTransferSolver* heat,
                std::unordered_map<int, const VariableNode*> alphaByDomain,
                std::unordered_map<int, const VariableNode*> youngByDomain,
                std::unordered_map<int, const VariableNode*> nuByDomain,
                Real tref)
                : heat_(heat), alphaByDomain_(std::move(alphaByDomain)), youngByDomain_(std::move(youngByDomain)), nuByDomain_(std::move(nuByDomain)), tref_(tref)
            {
                if (!heat_) {
                    MPFEM_THROW(ArgumentException, "ThermalExpansionStressNode requires heat solver.");
                }
            }

            VariableShape shape() const override { return VariableShape::Matrix; }
            std::pair<int, int> dimensions() const override { return {3, 3}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "ThermalExpansionStressNode requires ElementTransform in EvaluationContext.");
                }
                if (dest.size() != ctx.physicalPoints.size() * 9ull) {
                    MPFEM_THROW(ArgumentException, "ThermalExpansionStressNode destination size mismatch.");
                }

                const int domId = static_cast<int>(ctx.transform->attribute());
                const VariableNode* alphaNode = alphaByDomain_.at(domId);
                const VariableNode* eNode = youngByDomain_.at(domId);
                const VariableNode* nuNode = nuByDomain_.at(domId);

                const size_t pointCount = dest.size() / 9ull;
                for (size_t i = 0; i < pointCount; ++i) {
                    Vector3 refPoint = Vector3::Zero();
                    Vector3 physPoint = Vector3::Zero();
                    if (i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                        refPoint = ctx.referencePoints[i];
                    }
                    else {
                        const auto& ip = ctx.transform->integrationPoint();
                        refPoint = Vector3(ip.xi, ip.eta, ip.zeta);
                    }
                    if (i < ctx.physicalPoints.size()) {
                        physPoint = ctx.physicalPoints[i];
                    }
                    else {
                        const auto& ip = ctx.transform->integrationPoint();
                        ctx.transform->transform(ip, physPoint);
                    }

                    std::array<Vector3, 1> refPts {refPoint};
                    std::array<Vector3, 1> physPts {physPoint};
                    EvaluationContext one;
                    one.time = ctx.time;
                    one.domainId = domId;
                    one.elementId = ctx.elementId;
                    one.referencePoints = std::span<const Vector3>(refPts.data(), refPts.size());
                    one.physicalPoints = std::span<const Vector3>(physPts.data(), physPts.size());
                    one.transform = ctx.transform;

                    std::array<double, 9> alphaValues {};
                    std::array<double, 1> eValues {0.0};
                    std::array<double, 1> nuValues {0.0};
                    alphaNode->evaluateBatch(one, std::span<double>(alphaValues.data(), alphaValues.size()));
                    eNode->evaluateBatch(one, std::span<double>(eValues.data(), eValues.size()));
                    nuNode->evaluateBatch(one, std::span<double>(nuValues.data(), nuValues.size()));

                    Matrix3 alpha = Matrix3::Zero();
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            alpha(r, c) = static_cast<Real>(alphaValues[static_cast<size_t>(r * 3 + c)]);
                        }
                    }

                    const auto& ip = ctx.transform->integrationPoint();
                    const Real T = heat_->field().eval(ctx.transform->elementIndex(), &ip.xi);
                    const Real E_val = static_cast<Real>(eValues[0]);
                    const Real nu_val = static_cast<Real>(nuValues[0]);
                    const Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
                    const Real mu = E_val / (2.0 * (1.0 + nu_val));

                    const Matrix3 eps = alpha * (T - tref_);
                    const Matrix3 epsSym = 0.5 * (eps + eps.transpose());
                    Matrix3 sigma = 2.0 * mu * epsSym;
                    sigma.diagonal().array() += lambda * epsSym.trace();

                    const size_t base = i * 9ull;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            dest[base + static_cast<size_t>(r * 3 + c)] = sigma(r, c);
                        }
                    }
                }
            }

        private:
            const HeatTransferSolver* heat_ = nullptr;
            std::unordered_map<int, const VariableNode*> alphaByDomain_;
            std::unordered_map<int, const VariableNode*> youngByDomain_;
            std::unordered_map<int, const VariableNode*> nuByDomain_;
            Real tref_ = 0.0;
        };

    } // namespace

    namespace PhysicsProblemBuilder {
        void buildSolvers(Problem& problem);
        void setupCoupling(Problem& problem);
        void buildElectrostatics(Problem& problem, CaseDefinition::Physics& physics);
        void buildHeatTransfer(Problem& problem, CaseDefinition::Physics& physics);
        void buildStructural(Problem& problem, CaseDefinition::Physics& physics);

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

            // Register case variables to globalVariables_ before building expressions
            problem->registerCaseDefinitionVariables();

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
            auto& caseDef = problem.caseDef;

            // Build heat_transfer first because electrostatics material properties may depend on temperature "T"
            for (auto& [kind, physics] : caseDef.physics) {
                if (kind == "heat_transfer") {
                    buildHeatTransfer(problem, physics);
                    break;
                }
            }

            for (auto& [kind, physics] : caseDef.physics) {
                if (kind == "electrostatics") {
                    buildElectrostatics(problem, physics);
                    continue;
                }
                if (kind == "solid_mechanics") {
                    buildStructural(problem, physics);
                    continue;
                }
                // heat_transfer already handled above
            }

            if (problem.hasJouleHeating() || problem.hasThermalExpansion()) {
                setupCoupling(problem);
            }
        }

        void buildElectrostatics(Problem& problem, CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building electrostatics solver, order = " << physics.order;
            problem.electrostatics = std::make_unique<ElectrostaticsSolver>(physics.order);
            problem.electrostatics->setSolverConfig(std::move(physics.solver));

            double icValue = getInitialCondition(problem.caseDef, kPhysicsElectrostatics, 0.0);
            problem.electrostatics->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            // Register the electrostatics field as a DAG node for expression dependencies
            problem.globalVariables_.registerGridFunction("V", &problem.electrostatics->field());

            for (int domainId : problem.materials.domainIds()) {
                const VariableNode* sigma = requireDomainMatrixNode(
                    problem,
                    domainId,
                    kPropElectricConductivity);
                problem.electrostatics->setElectricalConductivity({domainId}, sigma);
            }

            for (const auto& bc : physics.boundaries) {
                if (bc.type == "Voltage") {
                    const VariableNode* voltage = makeScalarExpressionNode(problem, bc.parameters.at("value"));
                    problem.electrostatics->addVoltageBC(bc.ids, voltage);
                }
            }
        }

        void buildHeatTransfer(Problem& problem, CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building heat transfer solver, order = " << physics.order;
            problem.heatTransfer = std::make_unique<HeatTransferSolver>(physics.order);
            problem.heatTransfer->setSolverConfig(std::move(physics.solver));

            double icValue = getInitialCondition(problem.caseDef, kPhysicsHeatTransfer, 293.15);
            problem.heatTransfer->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            // Register the heat transfer field as a DAG node for expression dependencies
            problem.globalVariables_.registerGridFunction("T", &problem.heatTransfer->field());

            for (int domainId : problem.materials.domainIds()) {
                const VariableNode* k = requireDomainMatrixNode(
                    problem,
                    domainId,
                    kPropThermalConductivity);
                problem.heatTransfer->setThermalConductivity({domainId}, k);

                const VariableNode* rho = requireDomainScalarNode(problem, domainId, kPropDensity);
                const VariableNode* cp = requireDomainScalarNode(problem, domainId, kPropHeatCapacity);
                problem.heatTransfer->setMassProperties({domainId}, rho, cp);
            }

            for (const auto& bc : physics.boundaries) {
                if (bc.type == "Temperature") {
                    const VariableNode* temperature = makeScalarExpressionNode(problem, bc.parameters.at("value"));
                    problem.heatTransfer->addTemperatureBC(bc.ids, temperature);
                }
                else if (bc.type == "Convection") {
                    const VariableNode* h = makeScalarExpressionNode(problem, bc.parameters.at("h"));
                    const VariableNode* tinf = makeScalarExpressionNode(problem, bc.parameters.at("T_inf"));
                    problem.heatTransfer->addConvectionBC(bc.ids, h, tinf);
                }
            }
        }

        void buildStructural(Problem& problem, CaseDefinition::Physics& physics)
        {
            LOG_INFO << "Building structural solver, order = " << physics.order;
            problem.structural = std::make_unique<StructuralSolver>(physics.order);
            problem.structural->setSolverConfig(std::move(physics.solver));

            double icValue = getInitialCondition(problem.caseDef, kPhysicsSolidMechanics, 0.0);
            problem.structural->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            problem.globalVariables_.registerGridFunction("u", &problem.structural->field());

            for (int domainId : problem.materials.domainIds()) {
                const VariableNode* E = requireDomainScalarNode(problem, domainId, kPropYoungModulus);
                const VariableNode* nu = requireDomainScalarNode(problem, domainId, kPropPoissonRatio);
                problem.structural->addElasticity({domainId}, E, nu);
            }

            for (const auto& bc : physics.boundaries) {
                if (bc.type == "Fixed") {
                    problem.structural->addFixedDisplacementBC(bc.ids, Vector3::Zero());
                }
            }
        }

        void setupJouleHeating(Problem& problem, const CoupledPhysicsDefinition& cp)
        {
            const GridFunction* V_field = &problem.electrostatics->field();
            const std::set<int> activeDomains = cp.domainIds;
            const auto sigmaByDomain = collectDomainNodes(
                activeDomains,
                [&](int domainId) {
                    return requireDomainMatrixNode(problem, domainId, kPropElectricConductivity);
                });

            static std::atomic<std::uint64_t> id {0};
            std::string name = "JouleHeat_" + std::to_string(id++);
            auto jouleNode = std::make_unique<JouleHeatNode>(V_field, sigmaByDomain);
            problem.globalVariables_.adoptNode(std::move(jouleNode), name);
            const VariableNode* joule = problem.globalVariables_.get(name);
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
            const auto alphaByDomain = collectDomainNodes(
                activeDomains,
                [&](int domainId) {
                    return requireDomainMatrixNode(problem, domainId, kPropThermalExpansion);
                });
            const auto youngByDomain = collectDomainNodes(
                activeDomains,
                [&](int domainId) {
                    return requireDomainScalarNode(problem, domainId, kPropYoungModulus);
                });
            const auto nuByDomain = collectDomainNodes(
                activeDomains,
                [&](int domainId) {
                    return requireDomainScalarNode(problem, domainId, kPropPoissonRatio);
                });

            static std::atomic<std::uint64_t> id {0};
            std::string name = "ThermalExpansionStress_" + std::to_string(id++);
            auto stressNode = std::make_unique<ThermalExpansionStressNode>(
                problem.heatTransfer.get(),
                alphaByDomain,
                youngByDomain,
                nuByDomain,
                T_ref);
            problem.globalVariables_.adoptNode(std::move(stressNode), name);
            const VariableNode* stressCoef = problem.globalVariables_.get(name);
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

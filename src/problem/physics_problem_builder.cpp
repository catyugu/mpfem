#include "physics_problem_builder.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "expr/variable_graph.hpp"
#include "field/grid_function.hpp"
#include "io/problem_input_loader.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/field_values.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"

#include <atomic>

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

        Real getInitialCondition(const CaseDefinition& caseDef,
            std::string_view fieldKind,
            Real defaultVal)
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

        // =====================================================================
        // DomainMultiplexerProvider - selects child based on domain at runtime
        // =====================================================================
        class DomainMultiplexerProvider final : public VariableNode {
        public:
            explicit DomainMultiplexerProvider(TensorShape /*shape*/) { }

            void addDomain(int domainId, std::string targetName)
            {
                targetNames_[domainId] = std::move(targetName);
            }

            void resolve(const VariableManager& mgr) override
            {
                for (const auto& [did, name] : targetNames_) {
                    const VariableNode* child = mgr.get(name);
                    if (!child)
                        MPFEM_THROW(ArgumentException, "Unbound domain ref: " + name);
                    children_[did] = child;
                }
            }

            std::vector<const VariableNode*> getChildren() const override
            {
                std::vector<const VariableNode*> ret;
                for (const auto& [_, child] : children_) {
                    ret.push_back(child);
                }
                return ret;
            }

            std::uint64_t revision() const override
            {
                std::uint64_t rev = 0;
                for (const auto& [_, child] : children_) {
                    rev = std::max(rev, child->revision());
                }
                return rev;
            }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
            {
                auto it = children_.find(ctx.domainId);
                if (it == children_.end()) {
                    MPFEM_THROW(ArgumentException, "DomainMultiplexerProvider missing child for domain " + std::to_string(ctx.domainId));
                }
                it->second->evaluateBatch(ctx, dest);
            }

        private:
            std::unordered_map<int, std::string> targetNames_;
            std::unordered_map<int, const VariableNode*> children_;
        };

        // =====================================================================
        // GridFunctionValueProvider - evaluates GridFunction at quadrature points
        // =====================================================================
        class GridFunctionValueProvider final : public VariableNode {
        public:
            explicit GridFunctionValueProvider(const GridFunction* field)
                : field_(field) { }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Tensor::scalar(Real(0)));
                    return;
                }

                for (size_t i = 0; i < dest.size(); ++i) {
                    if (i >= ctx.transforms.size() || !ctx.transforms[i]) {
                        MPFEM_THROW(ArgumentException,
                            "GridFunctionValueProvider requires transforms in EvaluationContext.");
                    }
                    dest[i] = Tensor::scalar(field_->eval(ctx.elementId, *ctx.transforms[i]));
                }
            }

            std::uint64_t revision() const override
            {
                return field_ ? field_->revision() : 0;
            }

        private:
            const GridFunction* field_ = nullptr;
        };

        // =====================================================================
        // GridFunctionGradientProvider - evaluates gradient of GridFunction
        // =====================================================================
        class GridFunctionGradientProvider final : public VariableNode {
        public:
            explicit GridFunctionGradientProvider(const GridFunction* field)
                : field_(field) { }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Tensor::zero(TensorShape::vector(3)));
                    return;
                }

                if (ctx.referencePoints.size() < dest.size() || ctx.transforms.size() < dest.size()) {
                    MPFEM_THROW(ArgumentException,
                        "GridFunctionGradientProvider requires referencePoints and transforms in EvaluationContext.");
                }

                for (size_t i = 0; i < dest.size(); ++i) {
                    ElementTransform* trans = ctx.transforms[i];
                    if (!trans) {
                        MPFEM_THROW(ArgumentException,
                            "GridFunctionGradientProvider received null transform in EvaluationContext.");
                    }
                    Vector3 g = field_->gradient(ctx.elementId, *trans);
                    dest[i] = Tensor::vector(g);
                }
            }

            std::uint64_t revision() const override
            {
                return field_ ? field_->revision() : 0;
            }

        private:
            const GridFunction* field_ = nullptr;
        };

        // =====================================================================
        // Helper functions
        // =====================================================================

        const VariableNode* requireDomainPropertyNode(Problem& problem,
            std::string_view property,
            bool matrixProperty)
        {
            const std::string nodeName(property);
            if (const VariableNode* existing = problem.globalVariables_.get(nodeName)) {
                return existing;
            }

            TensorShape shape = matrixProperty ? TensorShape::matrix(3, 3) : TensorShape::scalar();
            auto selector = std::make_unique<DomainMultiplexerProvider>(shape);

            for (int domainId : problem.materials.domainIds()) {
                const std::string leafName = std::string(property) + "$domain_" + std::to_string(domainId);
                if (!problem.globalVariables_.get(leafName)) {
                    const std::string& expression = matrixProperty
                        ? problem.materials.matrixExpressionByDomain(domainId, property)
                        : problem.materials.scalarExpressionByDomain(domainId, property);
                    problem.globalVariables_.define(leafName, expression);
                }
                selector->addDomain(domainId, leafName);
            }

            problem.globalVariables_.bindNode(nodeName, std::move(selector));
            return problem.globalVariables_.get(nodeName);
        }

        const VariableNode* requireDomainMatrixNode(Problem& problem, std::string_view property)
        {
            return requireDomainPropertyNode(problem, property, true);
        }

        const VariableNode* requireDomainScalarNode(Problem& problem, std::string_view property)
        {
            return requireDomainPropertyNode(problem, property, false);
        }

        const VariableNode* makeScalarExpressionNode(Problem& problem, const std::string& expression)
        {
            static std::atomic<std::uint64_t> id {0};
            std::string name = "$expr_" + std::to_string(id++);
            problem.globalVariables_.define(name, expression);
            return problem.globalVariables_.get(name);
        }

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

            // ==========================================
            // KEY REFACTORING POINT: Build完毕后，统一执行compile()进行全树链接
            // ==========================================
            LOG_INFO << "Compiling variable expression ASTs...";
            problem->globalVariables_.compile();

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

            Real icValue = getInitialCondition(problem.caseDef, kPhysicsElectrostatics, 0.0);
            problem.electrostatics->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            // Register the electrostatics field as a DAG node for expression dependencies
            problem.globalVariables_.bindNode("V", std::make_unique<GridFunctionValueProvider>(&problem.electrostatics->field()));
            problem.globalVariables_.bindNode("grad_V", std::make_unique<GridFunctionGradientProvider>(&problem.electrostatics->field()));

            const VariableNode* sigma = requireDomainMatrixNode(problem, kPropElectricConductivity);
            for (int domainId : problem.materials.domainIds()) {
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

            Real icValue = getInitialCondition(problem.caseDef, kPhysicsHeatTransfer, 293.15);
            problem.heatTransfer->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            // Register the heat transfer field as a DAG node for expression dependencies
            problem.globalVariables_.bindNode("T", std::make_unique<GridFunctionValueProvider>(&problem.heatTransfer->field()));
            problem.globalVariables_.bindNode("grad(T)", std::make_unique<GridFunctionGradientProvider>(&problem.heatTransfer->field()));

            const VariableNode* k = requireDomainMatrixNode(problem, kPropThermalConductivity);
            (void)requireDomainScalarNode(problem, kPropDensity);
            (void)requireDomainScalarNode(problem, kPropHeatCapacity);

            problem.globalVariables_.define("thermal_mass", "density * heatcapacity");

            const VariableNode* rhoCpNode = problem.globalVariables_.get("thermal_mass");

            for (int domainId : problem.materials.domainIds()) {
                problem.heatTransfer->setThermalConductivity({domainId}, k);
                problem.heatTransfer->setMassProperties({domainId}, rhoCpNode);
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

            Real icValue = getInitialCondition(problem.caseDef, kPhysicsSolidMechanics, 0.0);
            problem.structural->initialize(*problem.mesh, problem.fieldValues, physics.order, icValue);

            const VariableNode* E = requireDomainScalarNode(problem, kPropYoungModulus);
            const VariableNode* nu = requireDomainScalarNode(problem, kPropPoissonRatio);
            for (int domainId : problem.materials.domainIds()) {
                problem.structural->addElasticity({domainId}, E, nu);
            }

            for (const auto& bc : physics.boundaries) {
                if (bc.type == "Fixed") {
                    problem.structural->addFixedDisplacementBC(bc.ids, problem.globalVariables_.get("[0, 0, 0]"));
                }
            }
        }

        void setupJouleHeating(Problem& problem, const CoupledPhysicsDefinition& cp)
        {
            const std::set<int> activeDomains = cp.domainIds;
            (void)requireDomainMatrixNode(problem, kPropElectricConductivity);
            problem.globalVariables_.define("JouleHeat", "dot(grad_V, electricconductivity * grad_V)");
            const VariableNode* joule = problem.globalVariables_.get("JouleHeat");
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
            (void)requireDomainMatrixNode(problem, kPropThermalExpansion);
            (void)requireDomainScalarNode(problem, kPropYoungModulus);
            (void)requireDomainScalarNode(problem, kPropPoissonRatio);

            problem.globalVariables_.define("lambda", "E*nu/((1+nu)*(1-2*nu))");
            problem.globalVariables_.define("mu", "E/(2*(1+nu))");
            problem.globalVariables_.define(
                "alpha_iso",
                "dot([1,1,1]^T, thermalexpansioncoefficient * [1,1,1]^T)/3.0");
            problem.globalVariables_.define("T_ref", std::to_string(T_ref));
            problem.globalVariables_.define("thermal_dT", "T - T_ref");
            problem.globalVariables_.define(
                "ThermalExpansionStress",
                "2*mu*sym([alpha_iso,0,0;0,alpha_iso,0;0,0,alpha_iso]*thermal_dT)"
                "+lambda*trace(sym([alpha_iso,0,0;0,alpha_iso,0;0,0,alpha_iso]*thermal_dT))*[1,0,0;0,1,0;0,0,1]");

            const VariableNode* stressCoef = problem.globalVariables_.get("ThermalExpansionStress");
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
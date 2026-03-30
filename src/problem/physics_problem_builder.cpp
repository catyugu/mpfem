#include "physics_problem_builder.hpp"
#include "problem.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"
#include "physics/field_values.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "io/mphtxt_reader.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/string_utils.hpp"
#include <cctype>
#include <functional>
#include <optional>
#include <unordered_set>

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

        bool isIdentifierStart(char c)
        {
            return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
        }

        bool isIdentifierChar(char c)
        {
            return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
        }

        bool isExponentIdentifier(std::string_view text, size_t index)
        {
            if (index == 0 || index + 1 >= text.size())
            {
                return false;
            }
            const char c = text[index];
            if (c != 'e' && c != 'E')
            {
                return false;
            }

            const char prev = text[index - 1];
            if (std::isdigit(static_cast<unsigned char>(prev)) == 0 && prev != '.')
            {
                return false;
            }

            const char next = text[index + 1];
            return std::isdigit(static_cast<unsigned char>(next)) != 0 || next == '+' || next == '-';
        }

        std::unordered_set<std::string> collectIdentifiers(std::string_view expr)
        {
            std::unordered_set<std::string> identifiers;
            size_t index = 0;
            while (index < expr.size())
            {
                const char c = expr[index];
                if (!isIdentifierStart(c) || isExponentIdentifier(expr, index))
                {
                    ++index;
                    continue;
                }

                const size_t begin = index;
                ++index;
                while (index < expr.size() && isIdentifierChar(expr[index]))
                {
                    ++index;
                }

                size_t probe = index;
                while (probe < expr.size() && std::isspace(static_cast<unsigned char>(expr[probe])) != 0)
                {
                    ++probe;
                }
                if (probe < expr.size() && expr[probe] == '(')
                {
                    continue;
                }

                identifiers.emplace(expr.substr(begin, index - begin));
            }

            return identifiers;
        }

        struct ExpressionSymbolUsage
        {
            bool useTime = false;
            bool useSpace = false;
            bool useTemperature = false;
            bool usePotential = false;
            std::vector<std::string> caseVariables;
        };

        ExpressionSymbolUsage detectExpressionSymbolUsage(
            const std::string &expr,
            const CaseDefinition &caseDef)
        {
            const auto ids = collectIdentifiers(expr);

            ExpressionSymbolUsage usage;
            usage.useTime = ids.count("t") > 0;
            const bool useX = ids.count("x") > 0;
            const bool useY = ids.count("y") > 0;
            const bool useZ = ids.count("z") > 0;
            usage.useSpace = useX || useY || useZ;
            usage.useTemperature = ids.count("T") > 0;
            usage.usePotential = ids.count("V") > 0;

            usage.caseVariables.reserve(caseDef.variableMap_.size());
            for (const auto &[name, _] : caseDef.variableMap_)
            {
                if (ids.count(name) > 0)
                {
                    usage.caseVariables.push_back(name);
                }
            }
            return usage;
        }

        class RuntimeExpressionContext
        {
        public:
            RuntimeExpressionContext(const CaseDefinition &caseDef, const ExpressionSymbolUsage &usage)
            {
                const size_t runtimeSymbolCount =
                    static_cast<size_t>(usage.useTime) +
                    (usage.useSpace ? 3u : 0u) +
                    static_cast<size_t>(usage.useTemperature) +
                    static_cast<size_t>(usage.usePotential);
                values_.reserve(usage.caseVariables.size() + runtimeSymbolCount);

                for (const std::string &name : usage.caseVariables)
                {
                    const auto it = caseDef.variableMap_.find(name);
                    if (it != caseDef.variableMap_.end())
                    {
                        addSymbol(it->first, it->second);
                    }
                }

                if (usage.useTime)
                {
                    t_ = addSymbol("t", 0.0);
                }
                if (usage.useSpace)
                {
                    x_ = addSymbol("x", 0.0);
                    y_ = addSymbol("y", 0.0);
                    z_ = addSymbol("z", 0.0);
                }
                if (usage.useTemperature)
                {
                    T_ = addSymbol("T", 293.15);
                }
                if (usage.usePotential)
                {
                    V_ = addSymbol("V", 0.0);
                }

                bindings_.reserve(values_.size());
                for (NamedValue &entry : values_)
                {
                    bindings_.push_back(ExpressionParser::VariableBinding{entry.name, &entry.value});
                }
            }

            void updateTime(Real time)
            {
                if (t_)
                {
                    *t_ = time;
                }
            }

            void updateSpace(ElementTransform &trans)
            {
                if (!x_ || !y_ || !z_)
                {
                    return;
                }
                Vector3 pos;
                trans.transform(trans.integrationPoint(), pos);
                *x_ = pos.x();
                *y_ = pos.y();
                *z_ = pos.z();
            }

            void updateTemperature(Real value)
            {
                if (T_)
                {
                    *T_ = value;
                }
            }

            void updatePotential(Real value)
            {
                if (V_)
                {
                    *V_ = value;
                }
            }

            const std::vector<ExpressionParser::VariableBinding> &bindings() const
            {
                return bindings_;
            }

        private:
            struct NamedValue
            {
                std::string name;
                double value = 0.0;
            };

            double *addSymbol(std::string name, double value)
            {
                for (NamedValue &entry : values_)
                {
                    if (entry.name == name)
                    {
                        return &entry.value;
                    }
                }

                values_.push_back(NamedValue{std::move(name), value});
                return &values_.back().value;
            }

            std::vector<NamedValue> values_;
            std::vector<ExpressionParser::VariableBinding> bindings_;
            double *t_ = nullptr;
            double *x_ = nullptr;
            double *y_ = nullptr;
            double *z_ = nullptr;
            double *T_ = nullptr;
            double *V_ = nullptr;
        };

        class CompiledScalarExpressionCoefficient : public Coefficient
        {
        public:
            using Updater = std::function<void(ElementTransform &, Real, RuntimeExpressionContext &)>;

            CompiledScalarExpressionCoefficient(ExpressionParser::ScalarProgram program,
                                               std::unique_ptr<RuntimeExpressionContext> context,
                                               Updater updater)
                : program_(std::move(program)),
                  context_(std::move(context)),
                  updater_(std::move(updater))
            {
            }

            void eval(ElementTransform &trans, Real &result, Real t = 0.0) const override
            {
                updater_(trans, t, *context_);
                result = program_.evaluate();
            }

        private:
            ExpressionParser::ScalarProgram program_;
            std::unique_ptr<RuntimeExpressionContext> context_;
            Updater updater_;
        };

        class CompiledMatrixExpressionCoefficient : public MatrixCoefficient
        {
        public:
            using Updater = std::function<void(ElementTransform &, Real, RuntimeExpressionContext &)>;

            CompiledMatrixExpressionCoefficient(ExpressionParser::MatrixProgram program,
                                               std::unique_ptr<RuntimeExpressionContext> context,
                                               Updater updater)
                : program_(std::move(program)),
                  context_(std::move(context)),
                  updater_(std::move(updater))
            {
            }

            void eval(ElementTransform &trans, Matrix3 &result, Real t = 0.0) const override
            {
                updater_(trans, t, *context_);
                result = program_.evaluate();
            }

        private:
            ExpressionParser::MatrixProgram program_;
            std::unique_ptr<RuntimeExpressionContext> context_;
            Updater updater_;
        };

        std::function<void(ElementTransform &, Real, RuntimeExpressionContext &)> makeUpdater(
            Problem &problem,
            const ExpressionSymbolUsage &usage)
        {
            return [&problem, usage](ElementTransform &trans, Real t, RuntimeExpressionContext &context)
            {
                if (usage.useTime)
                {
                    context.updateTime(t);
                }
                if (usage.useSpace)
                {
                    context.updateSpace(trans);
                }
                if (usage.useTemperature && problem.heatTransfer)
                {
                    const auto &ip = trans.integrationPoint();
                    const Real T = problem.heatTransfer->field().eval(trans.elementIndex(), &ip.xi);
                    context.updateTemperature(T);
                }
                if (usage.usePotential && problem.electrostatics)
                {
                    const auto &ip = trans.integrationPoint();
                    const Real V = problem.electrostatics->field().eval(trans.elementIndex(), &ip.xi);
                    context.updatePotential(V);
                }
            };
        }

        std::unique_ptr<Coefficient> makeScalarExpressionCoefficient(Problem &problem,
                                                                      const std::string &expression)
        {
            const auto usage = detectExpressionSymbolUsage(expression, problem.caseDef);
            auto context = std::make_unique<RuntimeExpressionContext>(problem.caseDef, usage);
            auto updater = makeUpdater(problem, usage);
            auto program = ExpressionParser::instance().compileScalar(expression, context->bindings());
            return std::make_unique<CompiledScalarExpressionCoefficient>(
                std::move(program), std::move(context), std::move(updater));
        }

        std::unique_ptr<MatrixCoefficient> makeMatrixExpressionCoefficient(Problem &problem,
                                                                            const std::string &expression)
        {
            const auto usage = detectExpressionSymbolUsage(expression, problem.caseDef);
            auto context = std::make_unique<RuntimeExpressionContext>(problem.caseDef, usage);
            auto updater = makeUpdater(problem, usage);
            auto program = ExpressionParser::instance().compileMatrix(expression, context->bindings());
            return std::make_unique<CompiledMatrixExpressionCoefficient>(
                std::move(program), std::move(context), std::move(updater));
        }

    } // namespace

    namespace PhysicsProblemBuilder
    {
        void buildSolvers(Problem &problem);
        void setupCoupling(Problem &problem);
        void buildElectrostatics(Problem &problem, const CaseDefinition::Physics &physics);
        void buildHeatTransfer(Problem &problem, const CaseDefinition::Physics &physics);
        void buildStructural(Problem &problem, const CaseDefinition::Physics &physics);

        std::unique_ptr<Problem> build(const std::string &caseDir)
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

            for (const auto &[domId, matTag] : problem.domainMaterial)
            {
                const auto *mat = problem.materials.getMaterial(matTag);
                if (!mat)
                    continue;

                if (!mat->hasMatrix("electricconductivity"))
                    continue;

                std::string key = "conductivity_" + std::to_string(domId);
                problem.setMatrixCoef(
                    key,
                    makeMatrixExpressionCoefficient(problem, mat->matrixExpression("electricconductivity")));

                problem.electrostatics->setElectricalConductivity({domId}, problem.getMatrixCoef(key));
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

            for (const auto &[domId, matTag] : problem.domainMaterial)
            {
                const auto *mat = problem.materials.getMaterial(matTag);
                if (!mat)
                    continue;

                if (mat->hasMatrix("thermalconductivity"))
                {
                    std::string key = "thermal_conductivity_" + std::to_string(domId);
                    problem.setMatrixCoef(
                        key,
                        makeMatrixExpressionCoefficient(problem, mat->matrixExpression("thermalconductivity")));
                    problem.heatTransfer->setThermalConductivity({domId}, problem.getMatrixCoef(key));
                }

                if (mat->hasScalar("density"))
                {
                    std::string key = "density_" + std::to_string(domId);
                    problem.setScalarCoef(
                        key,
                        makeScalarExpressionCoefficient(problem, mat->scalarExpression("density")));
                    problem.heatTransfer->setDensity({domId}, problem.getScalarCoef(key));
                }

                if (mat->hasScalar("heatcapacity"))
                {
                    std::string key = "heat_capacity_" + std::to_string(domId);
                    problem.setScalarCoef(
                        key,
                        makeScalarExpressionCoefficient(problem, mat->scalarExpression("heatcapacity")));
                    problem.heatTransfer->setSpecificHeat({domId}, problem.getScalarCoef(key));
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

            for (const auto &[domId, matTag] : problem.domainMaterial)
            {
                const auto *mat = problem.materials.getMaterial(matTag);
                if (!mat)
                    continue;

                if (!mat->hasScalar("E") || !mat->hasScalar("nu"))
                {
                    throw ArgumentException("Material missing required structural properties E/nu: " + matTag);
                }

                std::string eKey = "young_" + std::to_string(domId);
                std::string nuKey = "poisson_" + std::to_string(domId);
                problem.setScalarCoef(eKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("E")));
                problem.setScalarCoef(nuKey, makeScalarExpressionCoefficient(problem, mat->scalarExpression("nu")));
                problem.structural->setYoungModulus({domId}, problem.getScalarCoef(eKey));
                problem.structural->setPoissonRatio({domId}, problem.getScalarCoef(nuKey));
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

        void setupCoupling(Problem &problem)
        {
            // Setup coupled physics - conductivity is now handled via expressions in buildElectrostatics

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
                            Matrix3 sigma_mat;
                            sigma_coef->eval(trans, sigma_mat, t);
                            Vector3 g = V_field->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
                            // For anisotropic: Q = g^T * sigma * g
                            result = g.transpose() * sigma_mat * g;
                        });

                    DomainMappedScalarCoefficient jouleHeatMap;
                    jouleHeatMap.set({cp.domainIds.begin(), cp.domainIds.end()}, jouleHeat.get());
                    problem.setScalarCoef("jouleHeat", std::move(jouleHeat));
                    auto jouleHeatMapPtr = std::make_unique<DomainMappedScalarCoefficient>(std::move(jouleHeatMap));
                    problem.setScalarCoef("jouleHeatMap", std::move(jouleHeatMapPtr));
                    problem.heatTransfer->setHeatSource(problem.getScalarCoef("jouleHeatMap"));
                    LOG_INFO << "Joule heating domains: " << cp.domainIds.size() << " domains";
                }
                else if (cp.kind == "thermal_expansion")
                {
                    // Get reference temperature from solid_mechanics physics block
                    auto physicsIt = problem.caseDef.physics.find("solid_mechanics");
                    MPFEM_ASSERT(physicsIt != problem.caseDef.physics.end(),
                                 "solid_mechanics physics block not found for thermal expansion coupling");
                    Real T_ref = physicsIt->second.referenceTemperature;

                    for (int domId : cp.domainIds)
                    {
                        const auto domIt = problem.domainMaterial.find(domId);
                        if (domIt == problem.domainMaterial.end())
                            continue;

                        const auto *mat = problem.materials.getMaterial(domIt->second);
                        // Thermal expansion coefficient is optional - skip if not present
                        if (!mat || !mat->hasMatrix("thermalexpansioncoefficient"))
                        {
                            LOG_WARN << "Material for domain " << domId << " does not have thermal expansion coefficient, skipping coupling";
                            continue;
                        }

                        std::string alphaKey = "alphaThermal_" + std::to_string(domId);
                        problem.setMatrixCoef(
                            alphaKey,
                            makeMatrixExpressionCoefficient(problem, mat->matrixExpression("thermalexpansioncoefficient")));

                        const MatrixCoefficient *alphaCoef = problem.getMatrixCoef(alphaKey);
                        MPFEM_ASSERT(alphaCoef != nullptr, "Failed to build thermal expansion base coefficient.");

                        std::string key = "thermalExp_" + std::to_string(domId);
                        auto coef = std::make_unique<MatrixFunctionCoefficient>(
                            [&problem, alphaCoef, T_ref](ElementTransform &trans, Matrix3 &result, Real t)
                            {
                                Matrix3 alpha;
                                alphaCoef->eval(trans, alpha, t);

                                Real T = T_ref;
                                if (problem.heatTransfer)
                                {
                                    const auto &ip = trans.integrationPoint();
                                    T = problem.heatTransfer->field().eval(trans.elementIndex(), &ip.xi);
                                }
                                result = alpha * (T - T_ref);
                            });
                        problem.structural->setThermalExpansion({domId}, coef.get());
                        problem.setMatrixCoef(key, std::move(coef));
                    }
                    LOG_INFO << "Thermal expansion coupling enabled (T_ref = " << T_ref << " K)";
                }
            }
        }

    } // namespace PhysicsProblemBuilder

} // namespace mpfem

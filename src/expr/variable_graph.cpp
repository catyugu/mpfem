#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace mpfem {
    namespace {

        std::uint64_t nextProgramId()
        {
            static std::atomic<std::uint64_t> id {1};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

        class ConstantScalarNode final : public VariableNode {
        public:
            explicit ConstantScalarNode(Real value) : value_(value) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                const size_t n = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
                if (dest.size() != n) {
                    MPFEM_THROW(ArgumentException, "ConstantScalarNode evaluate destination size mismatch.");
                }
                for (size_t i = 0; i < n; ++i) {
                    dest[i] = value_;
                }
            }

            bool isConstant() const override { return true; }

        private:
            Real value_ = Real(0);
        };

        class GridFunctionNode final : public VariableNode {
        public:
            GridFunctionNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Real(0));
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionNode requires ElementTransform in EvaluationContext.");
                }
                const size_t n = ctx.physicalPoints.size();
                for (size_t i = 0; i < n; ++i) {
                    const Real* xi = nullptr;
                    if (i < ctx.referencePoints.size()) {
                        xi = &ctx.referencePoints[i].x();
                    }
                    else {
                        xi = &ctx.transform->integrationPoint().xi;
                    }
                    dest[i] = field_->eval(ctx.transform->elementIndex(), xi);
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            const GridFunction* field_ = nullptr;
        };

        class GridFunctionGradientNode final : public VariableNode {
        public:
            GridFunctionGradientNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            TensorShape shape() const override { return TensorShape::vector(3); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Real(0));
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionGradientNode requires ElementTransform in EvaluationContext.");
                }

                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n * 3ull) {
                    MPFEM_THROW(ArgumentException, "GridFunctionGradientNode evaluate destination size mismatch.");
                }

                for (size_t i = 0; i < n; ++i) {
                    const Real* xi = nullptr;
                    if (i < ctx.referencePoints.size()) {
                        xi = &ctx.referencePoints[i].x();
                        ctx.transform->setIntegrationPoint(xi);
                    }
                    else {
                        xi = &ctx.transform->integrationPoint().xi;
                    }

                    const Vector3 g = field_->gradient(ctx.transform->elementIndex(), xi, *ctx.transform);
                    const size_t base = i * 3ull;
                    dest[base] = g.x();
                    dest[base + 1] = g.y();
                    dest[base + 2] = g.z();
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            const GridFunction* field_ = nullptr;
        };

        class ExternalDataNode final : public VariableNode {
        public:
            using ValueExtractor = std::function<Real(const EvaluationContext&, size_t pointIndex)>;

            ExternalDataNode(std::string name, ValueExtractor extractor)
                : name_(std::move(name)), extractor_(std::move(extractor)) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                for (size_t i = 0; i < n; ++i) {
                    dest[i] = extractor_(ctx, i);
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            ValueExtractor extractor_;
        };

        class RuntimeExpressionNode final : public VariableNode {
        public:
            RuntimeExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), dependencies_(std::move(dependencies)), program_(std::move(program)), shape_(program_.shape()), id_(nextProgramId())
            {
            }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                const size_t valueSize = shape_.size();
                if (dest.size() != n * valueSize) {
                    MPFEM_THROW(ArgumentException, "RuntimeExpressionNode evaluate destination size mismatch.");
                }

                std::vector<std::vector<Real>> depValues(dependencies_.size());
                std::vector<size_t> depSizes(dependencies_.size(), 0);
                for (size_t d = 0; d < dependencies_.size(); ++d) {
                    depSizes[d] = dependencies_[d]->shape().size();
                    depValues[d].assign(n * depSizes[d], Real(0));
                    dependencies_[d]->evaluateBatch(ctx, std::span<Real>(depValues[d].data(), depValues[d].size()));
                }

                std::vector<TensorValue> inputValues(dependencies_.size());
                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    for (size_t d = 0; d < dependencies_.size(); ++d) {
                        const size_t base = i * depSizes[d];
                        if (depSizes[d] == 1) {
                            inputValues[d] = TensorValue::scalar(depValues[d][base]);
                            continue;
                        }
                        if (depSizes[d] == 3) {
                            inputValues[d] = TensorValue::vector(
                                depValues[d][base],
                                depValues[d][base + 1],
                                depValues[d][base + 2]);
                            continue;
                        }
                        if (depSizes[d] == 9) {
                            inputValues[d] = TensorValue::matrix3(
                                depValues[d][base + 0], depValues[d][base + 1], depValues[d][base + 2],
                                depValues[d][base + 3], depValues[d][base + 4], depValues[d][base + 5],
                                depValues[d][base + 6], depValues[d][base + 7], depValues[d][base + 8]);
                            continue;
                        }
                        MPFEM_THROW(ArgumentException, "RuntimeExpressionNode unsupported dependency shape size.");
                    }

                    const TensorValue exprResult = program_.evaluate(std::span<const TensorValue>(inputValues.data(), inputValues.size()));

                    if (shape_.isScalar()) {
                        dest[i] = exprResult.scalar();
                        continue;
                    }

                    if (shape_.isVector()) {
                        const size_t base = i * valueSize;
                        for (size_t c = 0; c < valueSize; ++c) {
                            dest[base + c] = exprResult[static_cast<int>(c)];
                        }
                        continue;
                    }

                    if (shape_.isMatrix()) {
                        const size_t base = i * valueSize;
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                dest[base + static_cast<size_t>(r * 3 + c)] = exprResult.at(r, c);
                            }
                        }
                        continue;
                    }

                    MPFEM_THROW(ArgumentException, "RuntimeExpressionNode unsupported expression shape.");
                }
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            TensorShape shape_;
            std::uint64_t id_ = 0;
        };

    } // namespace

    VariableManager::VariableManager()
    {
        // Register built-in spatial coordinates x, y, z and time t as ExternalDataNode
        nodes_["x"] = std::make_unique<ExternalDataNode>("x",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].x();
            });

        nodes_["y"] = std::make_unique<ExternalDataNode>("y",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].y();
            });

        nodes_["z"] = std::make_unique<ExternalDataNode>("z",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].z();
            });

        nodes_["t"] = std::make_unique<ExternalDataNode>("t",
            [](const EvaluationContext& ctx, size_t) -> Real {
                return ctx.time;
            });

        graphDirty_ = true;
    }

    void VariableManager::registerConstantExpression(std::string name, std::string expressionText)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expressionText);

        // If expression has no dependencies (pure constant), create ConstantScalarNode directly
        MPFEM_ASSERT(program.dependencies().empty(), "Expected constant expression to have no dependencies.");
        const std::array<TensorValue, 0> noInputs {};
        Real value = program.evaluate(std::span<const TensorValue>(noInputs.data(), noInputs.size())).scalar();
        nodes_[std::move(name)] = std::make_unique<ConstantScalarNode>(value);

        graphDirty_ = true;
    }

    void VariableManager::registerExpression(std::string name, std::string expression)
    {
        ExpressionParser parser;
        std::unordered_map<std::string, TensorShape> registeredShapes;
        registeredShapes.reserve(nodes_.size());
        for (const auto& [symbol, node] : nodes_) {
            if (node) {
                registeredShapes.emplace(symbol, node->shape());
            }
        }
        ExpressionParser::ExpressionProgram program = parser.compile(expression, registeredShapes);

        // Resolve all symbol dependencies from the node store
        std::vector<const VariableNode*> dependencies;
        dependencies.reserve(program.dependencies().size());

        for (const std::string& symbol : program.dependencies()) {
            if (nodes_.find(symbol) == nodes_.end()) {
                ensureGradientNode(symbol);
            }
            const auto it = nodes_.find(symbol);
            MPFEM_ASSERT(it != nodes_.end() && it->second != nullptr,
                "Unbound symbol in expression: " + symbol);
            dependencies.push_back(it->second.get());
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeExpressionNode>(
            std::move(expression),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerGridFunction(std::string name, const GridFunction* field)
    {
        const std::string key = name;
        nodes_[key] = std::make_unique<GridFunctionNode>(key, field);
        gridFunctions_[key] = field;
        graphDirty_ = true;
    }

    void VariableManager::ensureGradientNode(std::string_view symbol)
    {
        if (symbol.size() <= 6 || symbol.substr(0, 5) != "grad(" || symbol.back() != ')') {
            return;
        }

        const std::string base = std::string(symbol.substr(5, symbol.size() - 6));
        const auto fieldIt = gridFunctions_.find(base);
        if (fieldIt == gridFunctions_.end() || !fieldIt->second) {
            return;
        }

        const std::string gradName(symbol);
        if (nodes_.find(gradName) != nodes_.end()) {
            return;
        }
        nodes_[gradName] = std::make_unique<GridFunctionGradientNode>(gradName, fieldIt->second);
    }

    void VariableManager::registerExternalSource(std::string name,
        std::function<Real(const EvaluationContext&, size_t pointIndex)> extractor)
    {
        nodes_[std::move(name)] = std::make_unique<ExternalDataNode>(name, std::move(extractor));
        graphDirty_ = true;
    }

    void VariableManager::adoptNode(std::unique_ptr<VariableNode> node, std::string name)
    {
        nodes_[std::move(name)] = std::move(node);
        graphDirty_ = true;
    }

    const VariableNode* VariableManager::get(std::string_view name) const
    {
        const auto it = nodes_.find(std::string(name));
        if (it == nodes_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    void VariableManager::clearExecutionPlan()
    {
        executionPlan_.clear();
    }

    void VariableManager::dfsVisit(const VariableNode* node,
        std::unordered_map<const VariableNode*, int>& marks,
        std::vector<const VariableNode*>& ordered) const
    {
        auto it = marks.find(node);
        if (it != marks.end()) {
            if (it->second == 1) {
                MPFEM_THROW(ArgumentException, "Variable graph has cyclic dependencies.");
            }
            if (it->second == 2) {
                return;
            }
        }

        marks[node] = 1;
        const std::vector<const VariableNode*> deps = node->dependencies();
        for (const VariableNode* dep : deps) {
            if (dep) {
                dfsVisit(dep, marks, ordered);
            }
        }
        marks[node] = 2;
        ordered.push_back(node);
    }

    void VariableManager::compileGraph()
    {
        if (!graphDirty_) {
            return;
        }

        clearExecutionPlan();

        std::unordered_map<const VariableNode*, int> marks;
        marks.reserve(nodes_.size() * 2);

        std::vector<const VariableNode*> ordered;
        ordered.reserve(nodes_.size());

        for (const auto& [_, node] : nodes_) {
            if (node) {
                dfsVisit(node.get(), marks, ordered);
            }
        }

        executionPlan_ = std::move(ordered);
        graphDirty_ = false;
    }

} // namespace mpfem

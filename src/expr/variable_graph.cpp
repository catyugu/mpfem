#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"

#include <algorithm>
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

        // =============================================================================
        // Node implementations
        // Note: RuntimeExpressionNode uses local scratchpad vectors to avoid nesting issues
        // =============================================================================

        class ConstantNode final : public VariableNode {
        public:
            explicit ConstantNode(TensorValue value)
                : value_(std::move(value)), shape_(value_.shape()) { }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                (void)ctx; // Unused
                std::fill(dest.begin(), dest.end(), value_);
            }

            bool isConstant() const override { return true; }

        private:
            TensorValue value_;
            TensorShape shape_;
        };

        class GridFunctionNode final : public VariableNode {
        public:
            GridFunctionNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                const size_t n = dest.size();
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), TensorValue::scalar(Real(0)));
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionNode requires ElementTransform in EvaluationContext.");
                }
                for (size_t i = 0; i < n; ++i) {
                    const Real* xi = nullptr;
                    if (i < ctx.referencePoints.size()) {
                        xi = &ctx.referencePoints[i].x();
                    }
                    else {
                        xi = &ctx.transform->integrationPoint().xi;
                    }
                    dest[i] = TensorValue::scalar(field_->eval(ctx.transform->elementIndex(), xi));
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

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                const size_t n = dest.size();
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), TensorValue::zero(TensorShape::vector(3)));
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionGradientNode requires ElementTransform in EvaluationContext.");
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
                    dest[i] = TensorValue::vector(g.x(), g.y(), g.z());
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

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                const size_t n = dest.size();
                for (size_t i = 0; i < n; ++i) {
                    dest[i] = TensorValue::scalar(extractor_(ctx, i));
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            ValueExtractor extractor_;
        };

        // =============================================================================
        // Runtime expression node - original correct implementation
        // =============================================================================

        class RuntimeExpressionNode final : public VariableNode {
        public:
            RuntimeExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), dependencies_(std::move(dependencies)), program_(std::move(program)), shape_(program_.shape()), id_(nextProgramId())
            {
            }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                const size_t n = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
                if (dest.size() != n) {
                    MPFEM_THROW(ArgumentException, "RuntimeExpressionNode evaluate destination size mismatch.");
                }

                const size_t m = dependencies_.size();
                Workspace& workspace = workspaceFor(id_);

                // Keep workspace in-node to avoid per-batch heap allocations.
                if (workspace.scratchpad.size() != m * n) {
                    workspace.scratchpad.resize(m * n);
                }

                // Evaluate each dependency for all n points
                // Layout: [dep0_p0, dep0_p1, ..., dep0_p{n-1}, dep1_p0, dep1_p1, ...]
                for (size_t d = 0; d < m; ++d) {
                    std::span<TensorValue> depDest(&workspace.scratchpad[d * n], n);
                    dependencies_[d]->evaluateBatch(ctx, depDest);
                }

                if (workspace.pointInputs.size() != m) {
                    workspace.pointInputs.resize(m);
                }

                // Per-point expression evaluation.
                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    // Collect i-th value of each dependency into local buffer
                    for (size_t d = 0; d < m; ++d) {
                        workspace.pointInputs[d] = workspace.scratchpad[d * n + i];
                    }

                    dest[i] = program_.evaluate(
                        std::span<const TensorValue>(workspace.pointInputs.data(), m));
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return dependencies_; }

        private:
            struct Workspace {
                std::vector<TensorValue> scratchpad;
                std::vector<TensorValue> pointInputs;
            };

            static Workspace& workspaceFor(std::uint64_t nodeId)
            {
                thread_local std::unordered_map<std::uint64_t, Workspace> workspaceByNode;
                return workspaceByNode[nodeId];
            }

            std::string expression_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            TensorShape shape_;
            std::uint64_t id_ = 0;
        };

    } // namespace

    VariableManager::VariableManager()
    {
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

        MPFEM_ASSERT(program.dependencies().empty(), "Expected constant expression to have no dependencies.");

        // Directly get the full TensorValue (not limited to .scalar())
        const std::array<TensorValue, 0> noInputs {};
        TensorValue value = program.evaluate(std::span<const TensorValue>(noInputs.data(), noInputs.size()));

        nodes_[std::move(name)] = std::make_unique<ConstantNode>(std::move(value));

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

        // Pre-process: ensure gradient nodes exist for any grad(...) dependencies
        // before compile so inferShape can find them
        ExpressionParser::ExpressionProgram tempProgram = parser.compile(expression, registeredShapes);
        for (const std::string& symbol : tempProgram.dependencies()) {
            ensureGradientNode(symbol);
        }

        // Rebuild registeredShapes to include newly created gradient nodes
        registeredShapes.clear();
        registeredShapes.reserve(nodes_.size());
        for (const auto& [symbol, node] : nodes_) {
            if (node) {
                registeredShapes.emplace(symbol, node->shape());
            }
        }

        ExpressionParser::ExpressionProgram program = parser.compile(expression, registeredShapes);

        std::vector<const VariableNode*> dependencies;
        dependencies.reserve(program.dependencies().size());

        for (const std::string& symbol : program.dependencies()) {
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

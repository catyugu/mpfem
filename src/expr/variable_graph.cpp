#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"

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

        class RuntimeExpressionNode final : public VariableNode {
        public:
            RuntimeExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)),
                  dependencies_(std::move(dependencies)),
                  program_(std::move(program)),
                  shape_(program_.shape()),
                  id_(nextProgramId())
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

                if (workspace.scratchpad.size() != m * n) {
                    workspace.scratchpad.resize(m * n);
                }

                for (size_t d = 0; d < m; ++d) {
                    std::span<TensorValue> depDest(&workspace.scratchpad[d * n], n);
                    dependencies_[d]->evaluateBatch(ctx, depDest);
                }

                if (workspace.pointInputs.size() != m) {
                    workspace.pointInputs.resize(m);
                }

                for (size_t i = 0; i < n; ++i) {
                    for (size_t d = 0; d < m; ++d) {
                        workspace.pointInputs[d] = workspace.scratchpad[d * n + i];
                    }

                    dest[i] = program_.evaluate(std::span<const TensorValue>(workspace.pointInputs.data(), m));
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

        class ConstantNode final : public VariableNode {
        public:
            explicit ConstantNode(TensorValue value)
                : value_(std::move(value)), shape_(value_.shape())
            {
            }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                (void)ctx;
                std::fill(dest.begin(), dest.end(), value_);
            }

            bool isConstant() const override { return true; }

        private:
            TensorValue value_;
            TensorShape shape_;
        };

        class ExternalProviderNode final : public VariableNode {
        public:
            explicit ExternalProviderNode(std::unique_ptr<ExternalDataProvider> provider)
                : provider_(std::move(provider))
            {
                MPFEM_ASSERT(provider_ != nullptr, "External provider must not be null.");
            }

            TensorShape shape() const override { return provider_->shape(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                provider_->evaluateBatch(ctx, dest);
            }

        private:
            std::unique_ptr<ExternalDataProvider> provider_;
        };

        class PointScalarProvider final : public ExternalDataProvider {
        public:
            using Extractor = std::function<Real(const EvaluationContext&, size_t)>;

            explicit PointScalarProvider(Extractor extractor)
                : extractor_(std::move(extractor))
            {
            }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
            {
                for (size_t i = 0; i < dest.size(); ++i) {
                    dest[i] = TensorValue::scalar(extractor_(ctx, i));
                }
            }

        private:
            Extractor extractor_;
        };

        std::unique_ptr<VariableNode> makeExpressionNode(
            const std::string& expression,
            const std::unordered_map<std::string, std::unique_ptr<VariableNode>>& nodes)
        {
            ExpressionParser parser;
            std::unordered_map<std::string, TensorShape> registeredShapes;
            registeredShapes.reserve(nodes.size());
            for (const auto& [symbol, node] : nodes) {
                if (node) {
                    registeredShapes.emplace(symbol, node->shape());
                }
            }

            ExpressionParser::ExpressionProgram program = parser.compile(expression, registeredShapes);
            if (program.dependencies().empty()) {
                const std::array<TensorValue, 0> noInputs {};
                TensorValue value = program.evaluate(std::span<const TensorValue>(noInputs.data(), noInputs.size()));
                return std::make_unique<ConstantNode>(std::move(value));
            }

            std::vector<const VariableNode*> dependencies;
            dependencies.reserve(program.dependencies().size());
            for (const std::string& symbol : program.dependencies()) {
                const auto it = nodes.find(symbol);
                MPFEM_ASSERT(it != nodes.end() && it->second != nullptr,
                    "Unbound symbol in expression: " + symbol);
                dependencies.push_back(it->second.get());
            }

            return std::make_unique<RuntimeExpressionNode>(
                expression,
                std::move(dependencies),
                std::move(program));
        }

    } // namespace

    VariableManager::VariableManager()
    {
        bindExternal("x", std::make_unique<PointScalarProvider>(
                              [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                                  return ctx.physicalPoints[pointIndex].x();
                              }));

        bindExternal("y", std::make_unique<PointScalarProvider>(
                              [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                                  return ctx.physicalPoints[pointIndex].y();
                              }));

        bindExternal("z", std::make_unique<PointScalarProvider>(
                              [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                                  return ctx.physicalPoints[pointIndex].z();
                              }));

        bindExternal("t", std::make_unique<PointScalarProvider>(
                              [](const EvaluationContext& ctx, size_t) -> Real {
                                  return ctx.time;
                              }));

        graphDirty_ = true;
    }

    void VariableManager::define(std::string name, std::string expression)
    {
        nodes_[std::move(name)] = makeExpressionNode(expression, nodes_);
        graphDirty_ = true;
    }

    void VariableManager::bindExternal(std::string name, std::unique_ptr<ExternalDataProvider> provider)
    {
        MPFEM_ASSERT(provider != nullptr, "bindExternal requires non-null provider.");
        nodes_[std::move(name)] = std::make_unique<ExternalProviderNode>(std::move(provider));
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
        for (const VariableNode* dep : node->dependencies()) {
            if (dep) {
                dfsVisit(dep, marks, ordered);
            }
        }
        marks[node] = 2;
        ordered.push_back(node);
    }

    void VariableManager::compile()
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

    void VariableManager::evaluate(std::string_view name,
        const EvaluationContext& ctx,
        std::span<TensorValue> dest) const
    {
        const VariableNode* node = get(name);
        if (!node) {
            MPFEM_THROW(ArgumentException, "Variable not found: " + std::string(name));
        }
        node->evaluateBatch(ctx, dest);
    }

} // namespace mpfem

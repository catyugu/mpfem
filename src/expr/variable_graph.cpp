#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace mpfem {
    namespace {

        enum class BuiltInSymbolKind {
            None,
            Time,
            X,
            Y,
            Z,
        };

        struct RuntimeSymbolBinding {
            std::string symbol;
            BuiltInSymbolKind kind = BuiltInSymbolKind::None;
            const VariableNode* node = nullptr;
            GraphExternalSymbolResolver resolver;
            double constantValue = 0.0;
            bool hasConstant = false;
        };

        struct RuntimeSymbolConfig {
            std::vector<RuntimeSymbolBinding> bindings;
            std::vector<const VariableNode*> dependencies;
        };

        std::uint64_t nextProgramId()
        {
            static std::atomic<std::uint64_t> id {1};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

        BuiltInSymbolKind classifyBuiltInSymbol(std::string_view symbol)
        {
            if (symbol == "t") {
                return BuiltInSymbolKind::Time;
            }
            if (symbol == "x") {
                return BuiltInSymbolKind::X;
            }
            if (symbol == "y") {
                return BuiltInSymbolKind::Y;
            }
            if (symbol == "z") {
                return BuiltInSymbolKind::Z;
            }
            return BuiltInSymbolKind::None;
        }

        double readScalarNode(const VariableNode* node, const EvaluationContext& ctx)
        {
            if (!node || node->shape() != VariableShape::Scalar) {
                MPFEM_THROW(ArgumentException, "Expected scalar variable node.");
            }

            std::array<Vector3, 1> refPoints {};
            std::array<Vector3, 1> physPoints {};
            if (!ctx.referencePoints.empty()) {
                refPoints[0] = ctx.referencePoints.front();
            }
            if (!ctx.physicalPoints.empty()) {
                physPoints[0] = ctx.physicalPoints.front();
            }

            EvaluationContext onePointCtx = ctx;
            onePointCtx.referencePoints = std::span<const Vector3>(refPoints.data(), 1);
            onePointCtx.physicalPoints = std::span<const Vector3>(physPoints.data(), 1);

            std::array<double, 1> out {0.0};
            node->evaluateBatch(onePointCtx, std::span<double>(out.data(), out.size()));
            return out[0];
        }

        RuntimeSymbolConfig buildSymbolConfig(const std::vector<std::string>& symbols,
            const VariableManager::NodeStore& nodes,
            const GraphRuntimeResolvers& resolvers)
        {
            RuntimeSymbolConfig config;
            config.bindings.reserve(symbols.size());
            config.dependencies.reserve(symbols.size());

            for (const std::string& symbol : symbols) {
                RuntimeSymbolBinding binding;
                binding.symbol = symbol;
                binding.kind = classifyBuiltInSymbol(symbol);

                if (binding.kind != BuiltInSymbolKind::None) {
                    config.bindings.push_back(std::move(binding));
                    continue;
                }

                const auto it = nodes.find(symbol);
                if (it != nodes.end() && it->second && it->second->isConstant()) {
                    binding.hasConstant = true;
                    binding.constantValue = readScalarNode(it->second.get(), EvaluationContext {});
                    config.dependencies.push_back(it->second.get());
                    config.bindings.push_back(std::move(binding));
                    continue;
                }

                if (it != nodes.end() && it->second) {
                    binding.node = it->second.get();
                    config.dependencies.push_back(it->second.get());
                    config.bindings.push_back(std::move(binding));
                    continue;
                }

                if (resolvers.symbolBinder) {
                    binding.resolver = resolvers.symbolBinder(symbol);
                }

                config.bindings.push_back(std::move(binding));
            }

            return config;
        }

        std::vector<std::vector<double>> evaluateDependencyBlocks(const RuntimeSymbolConfig& config,
            const EvaluationContext& ctx,
            size_t numPoints)
        {
            std::vector<std::vector<double>> blocks(config.bindings.size());

            for (size_t slot = 0; slot < config.bindings.size(); ++slot) {
                const RuntimeSymbolBinding& binding = config.bindings[slot];
                if (!binding.node || binding.hasConstant) {
                    continue;
                }
                blocks[slot].assign(numPoints, 0.0);
                binding.node->evaluateBatch(ctx, std::span<double>(blocks[slot].data(), blocks[slot].size()));
            }

            return blocks;
        }

        double resolveSymbolValue(const RuntimeSymbolBinding& binding,
            const std::vector<std::vector<double>>& dependencyBlocks,
            const EvaluationContext& ctx,
            size_t slot,
            size_t pointIndex)
        {
            switch (binding.kind) {
            case BuiltInSymbolKind::Time:
                return ctx.time;
            case BuiltInSymbolKind::X:
                return ctx.physicalPoints[pointIndex].x();
            case BuiltInSymbolKind::Y:
                return ctx.physicalPoints[pointIndex].y();
            case BuiltInSymbolKind::Z:
                return ctx.physicalPoints[pointIndex].z();
            case BuiltInSymbolKind::None:
                break;
            }

            if (binding.hasConstant) {
                return binding.constantValue;
            }

            if (binding.node) {
                return dependencyBlocks[slot][pointIndex];
            }

            if (binding.resolver) {
                double resolved = 0.0;
                if (!binding.resolver(ctx, pointIndex, resolved)) {
                    MPFEM_THROW(ArgumentException, "Runtime symbol resolver failed for symbol: " + binding.symbol);
                }
                return resolved;
            }

            MPFEM_THROW(ArgumentException, "Unbound runtime symbol in expression: " + binding.symbol);
        }

        class ConstantScalarNode final : public VariableNode {
        public:
            explicit ConstantScalarNode(double value)
                : value_(value)
            {
            }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
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
            double value_ = 0.0;
        };

        class RuntimeScalarExpressionNode final : public VariableNode {
        public:
            RuntimeScalarExpressionNode(std::string expression,
                RuntimeSymbolConfig config,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), config_(std::move(config)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
            {
            }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n) {
                    MPFEM_THROW(ArgumentException, "RuntimeScalarExpressionNode evaluate destination size mismatch.");
                }

                const std::vector<std::vector<double>> dependencyBlocks = evaluateDependencyBlocks(config_, ctx, n);
                std::vector<double> inputValues(config_.bindings.size(), 0.0);

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    for (size_t slot = 0; slot < config_.bindings.size(); ++slot) {
                        inputValues[slot] = resolveSymbolValue(config_.bindings[slot], dependencyBlocks, ctx, slot, i);
                    }
                    ExprValue exprResult = program_.evaluate(std::span<const double>(inputValues.data(), inputValues.size()));
                    dest[i] = std::get<double>(exprResult);
                }
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            RuntimeSymbolConfig config_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            std::uint64_t id_ = 0;
        };

        class RuntimeMatrixExpressionNode final : public VariableNode {
        public:
            RuntimeMatrixExpressionNode(std::string expression,
                RuntimeSymbolConfig config,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), config_(std::move(config)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
            {
            }

            VariableShape shape() const override { return VariableShape::Matrix; }
            std::pair<int, int> dimensions() const override { return {3, 3}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n * 9ull) {
                    MPFEM_THROW(ArgumentException, "RuntimeMatrixExpressionNode evaluate destination size mismatch.");
                }

                const std::vector<std::vector<double>> dependencyBlocks = evaluateDependencyBlocks(config_, ctx, n);
                std::vector<double> inputValues(config_.bindings.size(), 0.0);

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    for (size_t slot = 0; slot < config_.bindings.size(); ++slot) {
                        inputValues[slot] = resolveSymbolValue(config_.bindings[slot], dependencyBlocks, ctx, slot, i);
                    }
                    const ExprValue exprResult = program_.evaluate(std::span<const double>(inputValues.data(), inputValues.size()));
                    const Matrix3 mat = std::get<Matrix3>(exprResult);
                    const size_t base = i * 9ull;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            dest[base + static_cast<size_t>(r * 3 + c)] = mat(r, c);
                        }
                    }
                }
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            RuntimeSymbolConfig config_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            std::uint64_t id_ = 0;
        };

    } // namespace

    void VariableManager::registerConstantExpression(std::string name, std::string expressionText)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expressionText);

        // If expression has no dependencies (pure constant), create ConstantScalarNode directly
        MPFEM_ASSERT(program.dependencies().empty(), "Expected constant expression to have no dependencies.");
        double value = std::get<double>(program.evaluate({}));
        nodes_[std::move(name)] = std::make_unique<ConstantScalarNode>(value);

        graphDirty_ = true;
    }

    void VariableManager::registerScalarExpression(std::string name,
        std::string expression,
        GraphRuntimeResolvers resolvers)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expression);
        RuntimeSymbolConfig config = buildSymbolConfig(program.dependencies(), nodes_, resolvers);
        std::vector<const VariableNode*> dependencies = config.dependencies;

        nodes_[std::move(name)] = std::make_unique<RuntimeScalarExpressionNode>(
            std::move(expression),
            std::move(config),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerMatrixExpression(std::string name,
        std::string expression,
        GraphRuntimeResolvers resolvers)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expression);
        RuntimeSymbolConfig config = buildSymbolConfig(program.dependencies(), nodes_, resolvers);
        std::vector<const VariableNode*> dependencies = config.dependencies;

        nodes_[std::move(name)] = std::make_unique<RuntimeMatrixExpressionNode>(
            std::move(expression),
            std::move(config),
            std::move(dependencies),
            std::move(program));
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

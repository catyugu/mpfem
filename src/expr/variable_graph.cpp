#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>


namespace mpfem {
    namespace {

        struct RuntimeSymbolConfig {
            std::vector<std::pair<std::string, double>> constants;
            std::vector<std::string> dynamicSymbols;
            std::vector<const VariableNode*> dependencies;
        };

        std::uint64_t nextProgramId()
        {
            static std::atomic<std::uint64_t> id {1};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

        enum class BuiltInSymbolKind {
            None,
            Time,
            X,
            Y,
            Z,
        };

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

        RuntimeSymbolConfig buildSymbolConfig(const ExpressionParser::ScalarProgram& program,
            const VariableManager::NodeStore& nodes)
        {
            RuntimeSymbolConfig config;
            std::unordered_set<std::string> seen;

            for (const std::string& symbol : program.dependencies()) {
                if (!seen.insert(symbol).second) {
                    continue;
                }

                const auto it = nodes.find(symbol);
                if (it != nodes.end() && it->second && it->second->isConstant()) {
                    const Real value = readScalarNode(it->second.get(), EvaluationContext {});
                    config.constants.emplace_back(symbol, value);
                    config.dependencies.push_back(it->second.get());
                    continue;
                }

                if (it != nodes.end() && it->second) {
                    config.dependencies.push_back(it->second.get());
                }
                else {
                    config.dynamicSymbols.push_back(symbol);
                }
            }

            return config;
        }

        std::unordered_map<std::string, double> makeValueMap(const RuntimeSymbolConfig& config,
            const EvaluationContext& ctx,
            size_t pointIndex,
            const std::vector<BuiltInSymbolKind>& kinds,
            const std::vector<GraphExternalSymbolResolver>& resolvers,
            const std::vector<std::string>& dynamicSymbols)
        {
            std::unordered_map<std::string, double> values;
            values.reserve(config.constants.size() + dynamicSymbols.size() + 4);

            for (const auto& [name, value] : config.constants) {
                values.emplace(name, value);
            }

            values.emplace("t", ctx.time);

            if (pointIndex < ctx.physicalPoints.size()) {
                const Vector3& p = ctx.physicalPoints[pointIndex];
                values.emplace("x", p.x());
                values.emplace("y", p.y());
                values.emplace("z", p.z());
            }

            for (size_t i = 0; i < dynamicSymbols.size(); ++i) {
                const std::string& symbol = dynamicSymbols[i];
                double value = 0.0;
                switch (kinds[i]) {
                case BuiltInSymbolKind::Time:
                    value = ctx.time;
                    break;
                case BuiltInSymbolKind::X:
                    value = ctx.physicalPoints[pointIndex].x();
                    break;
                case BuiltInSymbolKind::Y:
                    value = ctx.physicalPoints[pointIndex].y();
                    break;
                case BuiltInSymbolKind::Z:
                    value = ctx.physicalPoints[pointIndex].z();
                    break;
                case BuiltInSymbolKind::None:
                    if (i < resolvers.size() && resolvers[i]) {
                        double resolved = 0.0;
                        if (!resolvers[i](ctx, pointIndex, resolved)) {
                            MPFEM_THROW(ArgumentException, "Runtime symbol resolver failed for symbol: " + symbol);
                        }
                        value = resolved;
                    }
                    else {
                        MPFEM_THROW(ArgumentException, "Unbound runtime symbol in expression: " + symbol);
                    }
                    break;
                }
                values.insert_or_assign(symbol, value);
            }

            return values;
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
                std::vector<std::string> dynamicSymbols,
                std::vector<BuiltInSymbolKind> builtInKinds,
                std::vector<GraphExternalSymbolResolver> resolvers,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ScalarProgram program)
                : expression_(std::move(expression)), config_(std::move(config)), dynamicSymbols_(std::move(dynamicSymbols)), builtInKinds_(std::move(builtInKinds)), resolvers_(std::move(resolvers)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
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

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    const std::unordered_map<std::string, double> values = makeValueMap(config_, ctx, i, builtInKinds_, resolvers_, dynamicSymbols_);
                    dest[i] = program_.evaluate(values);
                }
            }

            bool isTimeDependent() const override
            {
                for (const BuiltInSymbolKind kind : builtInKinds_) {
                    if (kind == BuiltInSymbolKind::Time) {
                        return true;
                    }
                }
                return false;
            }

            bool isSpaceDependent() const override
            {
                for (const BuiltInSymbolKind kind : builtInKinds_) {
                    if (kind == BuiltInSymbolKind::X || kind == BuiltInSymbolKind::Y || kind == BuiltInSymbolKind::Z) {
                        return true;
                    }
                }
                return false;
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            RuntimeSymbolConfig config_;
            std::vector<std::string> dynamicSymbols_;
            std::vector<BuiltInSymbolKind> builtInKinds_;
            std::vector<GraphExternalSymbolResolver> resolvers_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ScalarProgram program_;
            std::uint64_t id_ = 0;
        };

        class RuntimeMatrixExpressionNode final : public VariableNode {
        public:
            RuntimeMatrixExpressionNode(std::string expression,
                RuntimeSymbolConfig config,
                std::vector<std::string> dynamicSymbols,
                std::vector<BuiltInSymbolKind> builtInKinds,
                std::vector<GraphExternalSymbolResolver> resolvers,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::MatrixProgram program)
                : expression_(std::move(expression)), config_(std::move(config)), dynamicSymbols_(std::move(dynamicSymbols)), builtInKinds_(std::move(builtInKinds)), resolvers_(std::move(resolvers)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
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

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    const std::unordered_map<std::string, double> values = makeValueMap(config_, ctx, i, builtInKinds_, resolvers_, dynamicSymbols_);
                    const Matrix3 mat = program_.evaluate(values);
                    const size_t base = i * 9ull;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            dest[base + static_cast<size_t>(r * 3 + c)] = mat(r, c);
                        }
                    }
                }
            }

            bool isTimeDependent() const override
            {
                for (const BuiltInSymbolKind kind : builtInKinds_) {
                    if (kind == BuiltInSymbolKind::Time) {
                        return true;
                    }
                }
                return false;
            }

            bool isSpaceDependent() const override
            {
                for (const BuiltInSymbolKind kind : builtInKinds_) {
                    if (kind == BuiltInSymbolKind::X || kind == BuiltInSymbolKind::Y || kind == BuiltInSymbolKind::Z) {
                        return true;
                    }
                }
                return false;
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            RuntimeSymbolConfig config_;
            std::vector<std::string> dynamicSymbols_;
            std::vector<BuiltInSymbolKind> builtInKinds_;
            std::vector<GraphExternalSymbolResolver> resolvers_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::MatrixProgram program_;
            std::uint64_t id_ = 0;
        };

    } // namespace

    void VariableManager::registerConstant(std::string name, double value)
    {
        nodes_[std::move(name)] = std::make_unique<ConstantScalarNode>(value);
        graphDirty_ = true;
    }

    void VariableManager::registerScalarExpression(std::string name,
        std::string expression,
        GraphRuntimeResolvers resolvers)
    {
        ExpressionParser parser;
        ExpressionParser::ScalarProgram program = parser.compileScalar(expression);
        RuntimeSymbolConfig config = buildSymbolConfig(program, nodes_);

        std::vector<std::string> dynamicSymbols = config.dynamicSymbols;
        std::vector<const VariableNode*> dependencies = config.dependencies;
        std::vector<BuiltInSymbolKind> builtInKinds;
        std::vector<GraphExternalSymbolResolver> boundResolvers;
        builtInKinds.reserve(dynamicSymbols.size());
        boundResolvers.reserve(dynamicSymbols.size());

        for (const std::string& symbol : dynamicSymbols) {
            const BuiltInSymbolKind kind = classifyBuiltInSymbol(symbol);
            builtInKinds.push_back(kind);
            if (kind != BuiltInSymbolKind::None) {
                boundResolvers.push_back({});
            }
            else if (resolvers.symbolBinder) {
                boundResolvers.push_back(resolvers.symbolBinder(symbol));
            }
            else {
                boundResolvers.push_back({});
            }
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeScalarExpressionNode>(
            std::move(expression),
            std::move(config),
            std::move(dynamicSymbols),
            std::move(builtInKinds),
            std::move(boundResolvers),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerMatrixExpression(std::string name,
        std::string expression,
        GraphRuntimeResolvers resolvers)
    {
        ExpressionParser parser;
        ExpressionParser::MatrixProgram program = parser.compileMatrix(expression);
        RuntimeSymbolConfig config;

        std::unordered_set<std::string> seen;

        for (const std::string& symbol : program.dependencies()) {
            if (!seen.insert(symbol).second) {
                continue;
            }

            const auto it = nodes_.find(symbol);
            if (it != nodes_.end() && it->second && it->second->isConstant()) {
                const Real value = readScalarNode(it->second.get(), EvaluationContext {});
                config.constants.emplace_back(symbol, value);
                config.dependencies.push_back(it->second.get());
                continue;
            }

            if (it != nodes_.end() && it->second) {
                config.dependencies.push_back(it->second.get());
            }
            else {
                config.dynamicSymbols.push_back(symbol);
            }
        }

        std::vector<std::string> dynamicSymbols = config.dynamicSymbols;
        std::vector<const VariableNode*> dependencies = config.dependencies;
        std::vector<BuiltInSymbolKind> builtInKinds;
        std::vector<GraphExternalSymbolResolver> boundResolvers;
        builtInKinds.reserve(dynamicSymbols.size());
        boundResolvers.reserve(dynamicSymbols.size());

        for (const std::string& symbol : dynamicSymbols) {
            const BuiltInSymbolKind kind = classifyBuiltInSymbol(symbol);
            builtInKinds.push_back(kind);
            if (kind != BuiltInSymbolKind::None) {
                boundResolvers.push_back({});
            }
            else if (resolvers.symbolBinder) {
                boundResolvers.push_back(resolvers.symbolBinder(symbol));
            }
            else {
                boundResolvers.push_back({});
            }
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeMatrixExpressionNode>(
            std::move(expression),
            std::move(config),
            std::move(dynamicSymbols),
            std::move(builtInKinds),
            std::move(boundResolvers),
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

#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "expr/symbol_scanner.hpp"
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
};

class RuntimeVariableStorage {
public:
    explicit RuntimeVariableStorage(const RuntimeSymbolConfig& config)
    {
        values_.reserve(config.constants.size() + config.dynamicSymbols.size());
        dynamicIndices_.reserve(config.dynamicSymbols.size());
        dynamicSymbols_.reserve(config.dynamicSymbols.size());

        for (const auto& [name, value] : config.constants) {
            addSymbol(name, value);
        }

        for (const std::string& name : config.dynamicSymbols) {
            const size_t index = addSymbol(name, 0.0);
            dynamicIndices_.push_back(index);
            dynamicSymbols_.push_back(name);
        }

        bindings_.reserve(values_.size());
        for (NamedValue& v : values_) {
            bindings_.push_back(ExpressionParser::VariableBinding{v.name, &v.value});
        }
    }

    void setDynamic(size_t index, double value)
    {
        MPFEM_ASSERT(index < dynamicIndices_.size(), "Dynamic symbol index is out of range.");
        values_[dynamicIndices_[index]].value = value;
    }

    const std::string& dynamicSymbol(size_t index) const
    {
        MPFEM_ASSERT(index < dynamicSymbols_.size(), "Dynamic symbol index is out of range.");
        return dynamicSymbols_[index];
    }

    size_t dynamicCount() const
    {
        return dynamicSymbols_.size();
    }

    const std::vector<ExpressionParser::VariableBinding>& bindings() const
    {
        return bindings_;
    }

private:
    struct NamedValue {
        std::string name;
        double value = 0.0;
    };

    size_t addSymbol(const std::string& name, double value)
    {
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i].name == name) {
                MPFEM_THROW(ArgumentException,
                            "Duplicated runtime symbol configuration for: " + name);
            }
        }

        values_.push_back(NamedValue{name, value});
        return values_.size() - 1;
    }

    std::vector<NamedValue> values_;
    std::vector<size_t> dynamicIndices_;
    std::vector<std::string> dynamicSymbols_;
    std::vector<ExpressionParser::VariableBinding> bindings_;
};

std::uint64_t nextProgramId()
{
    static std::atomic<std::uint64_t> id{1};
    return id.fetch_add(1, std::memory_order_relaxed);
}

bool resolveBuiltInSymbol(std::string_view symbol,
                          const EvaluationContext& ctx,
                          size_t pointIndex,
                          double& value)
{
    if (symbol == "t") {
        value = ctx.time;
        return true;
    }

    if (symbol != "x" && symbol != "y" && symbol != "z") {
        return false;
    }

    if (pointIndex >= ctx.physicalPoints.size()) {
        MPFEM_THROW(ArgumentException,
                    "Physical point index is out of range while resolving built-in coordinates.");
    }

    const Vector3& p = ctx.physicalPoints[pointIndex];
    if (symbol == "x") {
        value = p.x();
        return true;
    }
    if (symbol == "y") {
        value = p.y();
        return true;
    }

    value = p.z();
    return true;
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

class ConstantScalarNode final : public VariableNode {
public:
    explicit ConstantScalarNode(double value)
                : value_(value)
    {
    }

    VariableShape shape() const override { return VariableShape::Scalar; }
    std::pair<int, int> dimensions() const override { return {1, 1}; }

    void evaluate(const EvaluationContext& ctx, std::span<double> dest) const override
    {
        const size_t n = ctx.physicalPoints.size();
        if (dest.size() != n) {
            MPFEM_THROW(ArgumentException,
                        "ConstantScalarNode evaluate destination size mismatch.");
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
                                                                RuntimeSymbolConfig symbolConfig,
                                                                std::vector<std::string> dynamicSymbols,
                                                                std::vector<const VariableNode*> dependencies,
                                                                GraphRuntimeResolvers resolvers)
                : expression_(std::move(expression)),
            symbolConfig_(std::move(symbolConfig)),
          dynamicSymbols_(std::move(dynamicSymbols)),
          dependencies_(std::move(dependencies)),
          resolvers_(std::move(resolvers)),
            id_(nextProgramId())
    {
        boundResolvers_.reserve(dynamicSymbols_.size());
        builtInKinds_.reserve(dynamicSymbols_.size());
        for (const std::string& symbol : dynamicSymbols_) {
            const BuiltInSymbolKind kind = classifyBuiltInSymbol(symbol);
            builtInKinds_.push_back(kind);
            if (kind != BuiltInSymbolKind::None) {
                boundResolvers_.push_back({});
                continue;
            }
            if (!resolvers_.symbolBinder) {
                boundResolvers_.push_back({});
                continue;
            }
            boundResolvers_.push_back(resolvers_.symbolBinder(symbol));
        }
    }

    VariableShape shape() const override { return VariableShape::Scalar; }
    std::pair<int, int> dimensions() const override { return {1, 1}; }

    void evaluate(const EvaluationContext& ctx, std::span<double> dest) const override
    {
        ThreadState& state = getThreadState();
        const size_t n = ctx.physicalPoints.size();
        if (dest.size() != n) {
            MPFEM_THROW(ArgumentException,
                        "RuntimeScalarExpressionNode evaluate destination size mismatch.");
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

            for (size_t k = 0; k < state.variables.dynamicCount(); ++k) {
                double value = 0.0;
                if (!resolveDynamicValue(ctx, i, k, value)) {
                    const std::string& symbol = state.variables.dynamicSymbol(k);
                    MPFEM_THROW(ArgumentException,
                                "Runtime symbol resolver failed for symbol: " + symbol);
                }
                state.variables.setDynamic(k, value);
            }

            dest[i] = state.program.evaluate();
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
    bool resolveDynamicValue(const EvaluationContext& ctx,
                             size_t pointIndex,
                             size_t dynamicIndex,
                             double& value) const
    {
        const BuiltInSymbolKind kind = builtInKinds_[dynamicIndex];
        switch (kind) {
            case BuiltInSymbolKind::Time:
                value = ctx.time;
                return true;
            case BuiltInSymbolKind::X:
            case BuiltInSymbolKind::Y:
            case BuiltInSymbolKind::Z:
                return resolveBuiltInSymbol(dynamicSymbols_[dynamicIndex], ctx, pointIndex, value);
            case BuiltInSymbolKind::None:
                break;
        }

        const GraphExternalSymbolResolver& resolver = boundResolvers_[dynamicIndex];
        if (!resolver) {
            return false;
        }
        return resolver(ctx, pointIndex, value);
    }

    struct ThreadState {
        RuntimeVariableStorage variables;
        ExpressionParser::ScalarProgram program;

        ThreadState(const std::string& expression, const RuntimeSymbolConfig& config)
            : variables(config),
              program(ExpressionParser().compileScalar(expression, variables.bindings()))
        {
        }
    };

    ThreadState& getThreadState() const
    {
        static thread_local std::unordered_map<std::uint64_t, ThreadState> cache;

        auto it = cache.find(id_);
        if (it != cache.end()) {
            return it->second;
        }

        auto inserted = cache.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id_),
            std::forward_as_tuple(expression_, symbolConfig_));
        return inserted.first->second;
    }

    std::string expression_;
    RuntimeSymbolConfig symbolConfig_;
    std::vector<std::string> dynamicSymbols_;
    std::vector<BuiltInSymbolKind> builtInKinds_;
    std::vector<GraphExternalSymbolResolver> boundResolvers_;
    std::vector<const VariableNode*> dependencies_;
    GraphRuntimeResolvers resolvers_;
    std::uint64_t id_ = 0;
};

class RuntimeMatrixExpressionNode final : public VariableNode {
public:
        RuntimeMatrixExpressionNode(std::string expression,
                                                                RuntimeSymbolConfig symbolConfig,
                                                                std::vector<std::string> dynamicSymbols,
                                                                std::vector<const VariableNode*> dependencies,
                                                                GraphRuntimeResolvers resolvers)
                : expression_(std::move(expression)),
            symbolConfig_(std::move(symbolConfig)),
          dynamicSymbols_(std::move(dynamicSymbols)),
          dependencies_(std::move(dependencies)),
          resolvers_(std::move(resolvers)),
            id_(nextProgramId())
    {
        boundResolvers_.reserve(dynamicSymbols_.size());
        builtInKinds_.reserve(dynamicSymbols_.size());
        for (const std::string& symbol : dynamicSymbols_) {
            const BuiltInSymbolKind kind = classifyBuiltInSymbol(symbol);
            builtInKinds_.push_back(kind);
            if (kind != BuiltInSymbolKind::None) {
                boundResolvers_.push_back({});
                continue;
            }
            if (!resolvers_.symbolBinder) {
                boundResolvers_.push_back({});
                continue;
            }
            boundResolvers_.push_back(resolvers_.symbolBinder(symbol));
        }
    }

    VariableShape shape() const override { return VariableShape::Matrix; }
    std::pair<int, int> dimensions() const override { return {3, 3}; }

    void evaluate(const EvaluationContext& ctx, std::span<double> dest) const override
    {
        ThreadState& state = getThreadState();
        const size_t n = ctx.physicalPoints.size();
        if (dest.size() != n * 9ull) {
            MPFEM_THROW(ArgumentException,
                        "RuntimeMatrixExpressionNode evaluate destination size mismatch.");
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

            for (size_t k = 0; k < state.variables.dynamicCount(); ++k) {
                double value = 0.0;
                if (!resolveDynamicValue(ctx, i, k, value)) {
                    const std::string& symbol = state.variables.dynamicSymbol(k);
                    MPFEM_THROW(ArgumentException,
                                "Runtime symbol resolver failed for symbol: " + symbol);
                }
                state.variables.setDynamic(k, value);
            }

            const Matrix3 mat = state.program.evaluate();
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
    bool resolveDynamicValue(const EvaluationContext& ctx,
                             size_t pointIndex,
                             size_t dynamicIndex,
                             double& value) const
    {
        const BuiltInSymbolKind kind = builtInKinds_[dynamicIndex];
        switch (kind) {
            case BuiltInSymbolKind::Time:
                value = ctx.time;
                return true;
            case BuiltInSymbolKind::X:
            case BuiltInSymbolKind::Y:
            case BuiltInSymbolKind::Z:
                return resolveBuiltInSymbol(dynamicSymbols_[dynamicIndex], ctx, pointIndex, value);
            case BuiltInSymbolKind::None:
                break;
        }

        const GraphExternalSymbolResolver& resolver = boundResolvers_[dynamicIndex];
        if (!resolver) {
            return false;
        }
        return resolver(ctx, pointIndex, value);
    }

    struct ThreadState {
        RuntimeVariableStorage variables;
        ExpressionParser::MatrixProgram program;

        ThreadState(const std::string& expression, const RuntimeSymbolConfig& config)
            : variables(config),
              program(ExpressionParser().compileMatrix(expression, variables.bindings()))
        {
        }
    };

    ThreadState& getThreadState() const
    {
        static thread_local std::unordered_map<std::uint64_t, ThreadState> cache;

        auto it = cache.find(id_);
        if (it != cache.end()) {
            return it->second;
        }

        auto inserted = cache.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id_),
            std::forward_as_tuple(expression_, symbolConfig_));
        return inserted.first->second;
    }

    std::string expression_;
    RuntimeSymbolConfig symbolConfig_;
    std::vector<std::string> dynamicSymbols_;
    std::vector<BuiltInSymbolKind> builtInKinds_;
    std::vector<GraphExternalSymbolResolver> boundResolvers_;
    std::vector<const VariableNode*> dependencies_;
    GraphRuntimeResolvers resolvers_;
    std::uint64_t id_ = 0;
};

RuntimeSymbolConfig buildSymbolConfig(const std::string& expression,
                                      const VariableManager::NodeStore& nodes,
                                      std::vector<std::string>& dynamicSymbols,
                                      std::vector<const VariableNode*>& dependencies)
{
    RuntimeSymbolConfig config;
    const std::vector<std::string> symbols = collectExpressionSymbols(expression);

    dynamicSymbols.clear();
    dependencies.clear();

    std::unordered_set<std::string> seen;
    seen.reserve(symbols.size());

    for (const std::string& symbol : symbols) {
        if (!seen.insert(symbol).second) {
            continue;
        }

        const auto it = nodes.find(symbol);
        if (it != nodes.end() && it->second && it->second->isConstant()) {
            EvaluationContext emptyCtx;
            std::array<Vector3, 1> point{Vector3::Zero()};
            emptyCtx.physicalPoints = std::span<const Vector3>(point.data(), 1);
            std::array<double, 1> value{0.0};
            it->second->evaluate(emptyCtx, std::span<double>(value.data(), 1));
            config.constants.emplace_back(symbol, value[0]);
            dependencies.push_back(it->second.get());
            continue;
        }

        dynamicSymbols.push_back(symbol);
        if (it != nodes.end() && it->second) {
            dependencies.push_back(it->second.get());
        }
    }

    config.dynamicSymbols = dynamicSymbols;
    return config;
}

}  // namespace

void VariableManager::registerConstant(std::string name, double value)
{
    nodes_[std::move(name)] = std::make_unique<ConstantScalarNode>(value);
    graphDirty_ = true;
}

void VariableManager::registerScalarExpression(std::string name,
                                               std::string expression,
                                               GraphRuntimeResolvers resolvers)
{
    std::vector<std::string> dynamicSymbols;
    std::vector<const VariableNode*> dependencies;
    RuntimeSymbolConfig config = buildSymbolConfig(expression, nodes_, dynamicSymbols, dependencies);

    nodes_[std::move(name)] = std::make_unique<RuntimeScalarExpressionNode>(
        std::move(expression),
        std::move(config),
        std::move(dynamicSymbols),
        std::move(dependencies),
        std::move(resolvers));
    graphDirty_ = true;
}

void VariableManager::registerMatrixExpression(std::string name,
                                               std::string expression,
                                               GraphRuntimeResolvers resolvers)
{
    std::vector<std::string> dynamicSymbols;
    std::vector<const VariableNode*> dependencies;
    RuntimeSymbolConfig config = buildSymbolConfig(expression, nodes_, dynamicSymbols, dependencies);

    nodes_[std::move(name)] = std::make_unique<RuntimeMatrixExpressionNode>(
        std::move(expression),
        std::move(config),
        std::move(dynamicSymbols),
        std::move(dependencies),
        std::move(resolvers));
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

}  // namespace mpfem

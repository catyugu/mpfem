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

    // =========================================================================
    // Built-in VariableNode implementations for x, y, z, t coordinates and time
    // =========================================================================

    class PointScalarNode final : public VariableNode {
    public:
        using Extractor = Real (*)(const Vector3&);

        explicit PointScalarNode(Extractor extractor) : extractor_(extractor) { }

        void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
        {
            for (size_t i = 0; i < dest.size(); ++i) {
                dest[i] = Tensor::scalar(extractor_(ctx.physicalPoints[i]));
            }
        }

    private:
        Extractor extractor_;
    };

    class TimeNode final : public VariableNode {
    public:
        void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
        {
            for (auto& v : dest)
                v = Tensor::scalar(ctx.time);
        }
    };

    // =========================================================================
    // VariableManager implementation
    // =========================================================================

    VariableManager::VariableManager()
    {
        // Bind built-in spatial coordinate variables
        bindNode("x", std::make_unique<PointScalarNode>([](const Vector3& p) { return p.x(); }));
        bindNode("y", std::make_unique<PointScalarNode>([](const Vector3& p) { return p.y(); }));
        bindNode("z", std::make_unique<PointScalarNode>([](const Vector3& p) { return p.z(); }));
        bindNode("t", std::make_unique<TimeNode>());
    }

    void VariableManager::define(std::string name, const std::string& expression)
    {
        // Parse expression into AST tree - VariableRefNode nodes are unresolved at this point
        nodes_[std::move(name)] = ExpressionParser::parse(expression);
        isCompiled_ = false;
    }

    void VariableManager::bindNode(std::string name, std::unique_ptr<VariableNode> node)
    {
        MPFEM_ASSERT(node != nullptr, "bindNode requires non-null node.");
        nodes_[std::move(name)] = std::move(node);
        isCompiled_ = false;
    }

    const VariableNode* VariableManager::get(std::string_view name) const
    {
        auto it = nodes_.find(std::string(name));
        return it != nodes_.end() ? it->second.get() : nullptr;
    }

    void VariableManager::compile()
    {
        if (isCompiled_)
            return;

        // Phase 1: Resolve all VariableRefNode references to actual pointers
        for (const auto& [name, node] : nodes_) {
            (void)name; // suppress unused warning
            node->resolve(*this);
        }

        // Phase 2: Check for cyclic dependencies
        checkCycles();

        isCompiled_ = true;
    }

    void VariableManager::evaluate(std::string_view name,
        const EvaluationContext& ctx,
        std::span<Tensor> dest) const
    {
        MPFEM_ASSERT(isCompiled_, "VariableManager must be compiled before evaluation.");
        const VariableNode* node = get(name);
        if (!node) {
            MPFEM_THROW(ArgumentException, "Variable not found: " + std::string(name));
        }
        node->evaluateBatch(ctx, dest);
    }

    void VariableManager::checkCycles() const
    {
        std::unordered_map<const VariableNode*, int> state;
        for (const auto& [name, node] : nodes_) {
            (void)name;
            if (state[node.get()] == 0) {
                dfsVisit(node.get(), state);
            }
        }
    }

    void VariableManager::dfsVisit(const VariableNode* node,
        std::unordered_map<const VariableNode*, int>& state) const
    {
        state[node] = 1; // visiting

        for (const VariableNode* child : node->getChildren()) {
            if (!child)
                continue;
            if (state[child] == 1) {
                MPFEM_THROW(ArgumentException, "Cyclic dependency detected in variable expressions.");
            }
            if (state[child] == 0) {
                dfsVisit(child, state);
            }
        }

        state[node] = 2; // visited
    }

} // namespace mpfem
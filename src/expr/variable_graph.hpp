#ifndef MPFEM_EXPR_VARIABLE_GRAPH_HPP
#define MPFEM_EXPR_VARIABLE_GRAPH_HPP

#include "core/types.hpp"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpfem {

    class ElementTransform;

    enum class VariableShape {
        Scalar,
        Vector,
        Matrix,
    };

    struct EvaluationContext {
        double time = 0.0;
        int domainId = -1;
        Index elementId = InvalidIndex;
        std::span<const Vector3> physicalPoints;
        std::span<const Vector3> referencePoints;
        ElementTransform* transform = nullptr;
    };

    using GraphExternalSymbolResolver = std::function<bool(const EvaluationContext&, size_t, double&)>;

    using GraphExternalSymbolBinder = std::function<GraphExternalSymbolResolver(std::string_view)>;

    struct GraphRuntimeResolvers {
        GraphExternalSymbolBinder symbolBinder;
    };

    class VariableNode {
    public:
        virtual ~VariableNode() = default;

        virtual VariableShape shape() const = 0;
        virtual std::pair<int, int> dimensions() const = 0;
        virtual void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const = 0;
        virtual bool isConstant() const { return false; }
        virtual std::vector<const VariableNode*> dependencies() const { return {}; }
    };

    class VariableManager {
    public:
        using NodeStore = std::unordered_map<std::string, std::unique_ptr<VariableNode>>;

        VariableManager() = default;

        /**
         * @brief Register a constant expression.
         * @details If expression has no dependencies (pure constant), creates ConstantScalarNode.
         *         Otherwise creates RuntimeScalarExpressionNode.
         */
        void registerConstantExpression(std::string name, std::string expressionText);

        void registerScalarExpression(std::string name,
            std::string expression,
            GraphRuntimeResolvers resolvers = {});

        void registerMatrixExpression(std::string name,
            std::string expression,
            GraphRuntimeResolvers resolvers = {});

        const VariableNode* get(std::string_view name) const;

        void compileGraph();

    private:
        void clearExecutionPlan();
        void dfsVisit(const VariableNode* node,
            std::unordered_map<const VariableNode*, int>& marks,
            std::vector<const VariableNode*>& ordered) const;

        NodeStore nodes_;
        std::vector<const VariableNode*> executionPlan_;
        bool graphDirty_ = true;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_VARIABLE_GRAPH_HPP

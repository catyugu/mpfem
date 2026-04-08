#ifndef MPFEM_EXPR_VARIABLE_GRAPH_HPP
#define MPFEM_EXPR_VARIABLE_GRAPH_HPP

#include "core/tensor_shape.hpp"
#include "core/tensor_value.hpp"
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
    class GridFunction;

    struct EvaluationContext {
        Real time = Real(0);
        int domainId = -1;
        Index elementId = InvalidIndex;
        std::span<const Vector3> physicalPoints;
        std::span<const Vector3> referencePoints;
        ElementTransform* transform = nullptr;
    };

    class VariableNode {
    public:
        virtual ~VariableNode() = default;

        /// 返回张量形状，标量=empty/{}, 向量={n}, 矩阵={3,3}
        virtual TensorShape shape() const = 0;

        /// 批量求值：计算该节点在每个物理点的值
        /// @param ctx 评估上下文
        /// @param dest 输出缓冲区，大小 = physicalPoints.size()
        virtual void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const = 0;

        virtual bool isConstant() const { return false; }
        virtual std::vector<const VariableNode*> dependencies() const { return {}; }
    };

    class VariableManager {
    public:
        using NodeStore = std::unordered_map<std::string, std::unique_ptr<VariableNode>>;

        VariableManager();

        ~VariableManager() = default;

        /**
         * @brief Register a constant expression.
         * @details If expression has no dependencies (pure constant), creates ConstantScalarNode.
         *         Otherwise creates RuntimeExpressionNode.
         */
        void registerConstantExpression(std::string name, std::string expressionText);

        /**
         * @brief Register an expression (scalar, vector, or matrix).
         * @details Infers the shape from the expression itself via ExpressionProgram::shape().
         */
        void registerExpression(std::string name, std::string expression);

        const VariableNode* get(std::string_view name) const;

        void registerGridFunction(std::string name, const GridFunction* field);

        void registerExternalSource(std::string name,
            std::function<Real(const EvaluationContext&, size_t pointIndex)> extractor);

        void adoptNode(std::unique_ptr<VariableNode> node, std::string name);

        void compileGraph();

    private:
        void ensureGradientNode(std::string_view symbol);
        void clearExecutionPlan();
        void dfsVisit(const VariableNode* node,
            std::unordered_map<const VariableNode*, int>& marks,
            std::vector<const VariableNode*>& ordered) const;

        NodeStore nodes_;
        std::unordered_map<std::string, const GridFunction*> gridFunctions_;
        std::vector<const VariableNode*> executionPlan_;
        bool graphDirty_ = true;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_VARIABLE_GRAPH_HPP

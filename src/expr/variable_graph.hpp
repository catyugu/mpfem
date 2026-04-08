#ifndef MPFEM_EXPR_VARIABLE_GRAPH_HPP
#define MPFEM_EXPR_VARIABLE_GRAPH_HPP

#include "expr/evaluation_context.hpp"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpfem {

    class VariableNode {
    public:
        virtual ~VariableNode() = default;

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

        void define(std::string name, std::string expression);

        void bindNode(std::string name, std::unique_ptr<VariableNode> node);

        const VariableNode* get(std::string_view name) const;

        void compile();

        void evaluate(std::string_view name,
            const EvaluationContext& ctx,
            std::span<TensorValue> dest) const;

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

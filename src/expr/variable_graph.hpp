#ifndef MPFEM_EXPR_VARIABLE_GRAPH_HPP
#define MPFEM_EXPR_VARIABLE_GRAPH_HPP

#include "expr/evaluation_context.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpfem {

    class VariableManager;

    /**
     * @brief 统一的表达式节点接口（即是变量，也是 AST 节点）
     */
    class VariableNode {
    public:
        virtual ~VariableNode() = default;

        /// 批量求值：计算该节点在每个物理点的值
        /// @param ctx 评估上下文
        /// @param dest 输出缓冲区，大小 = referencePoints.size()
        virtual void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const = 0;

        /// 编译期链接：子类应在此处向 Manager 查找未解析的依赖
        virtual void resolve(const VariableManager& mgr) { (void)mgr; }

        /// 获取子节点（用于检测循环依赖）
        virtual std::vector<const VariableNode*> getChildren() const { return {}; }

        virtual bool isConstant() const { return false; }

        virtual std::uint64_t revision() const { return 0; }
    };

    /**
     * @brief 纯粹的变量与 AST 树管理器
     * 不再做拓扑排序执行计划，仅负责存储节点、链接 AST 以及环检测。
     */
    class VariableManager {
    public:
        VariableManager();
        ~VariableManager() = default;

        /// 定义一个字符串表达式（立刻解析为 AST，但不立即链接）
        void define(std::string name, const std::string& expression);

        /// 绑定一个自定义节点（例如 GridFunction 提供者）
        void bindNode(std::string name, std::unique_ptr<VariableNode> node);

        /// 获取节点（如果在 compile 前调用，可能包含未解析的引用）
        const VariableNode* get(std::string_view name) const;

        /// 编译阶段：链接所有 VariableRef，并检查循环依赖
        void compile();

        /// 直接对指定变量求值
        void evaluate(std::string_view name,
            const EvaluationContext& ctx,
            std::span<Tensor> dest) const;

    private:
        void checkCycles() const;
        void dfsVisit(const VariableNode* node,
            std::unordered_map<const VariableNode*, int>& state) const;

        std::unordered_map<std::string, std::unique_ptr<VariableNode>> nodes_;
        bool isCompiled_ = false;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_VARIABLE_GRAPH_HPP
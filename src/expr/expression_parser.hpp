#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include <memory>
#include <string>

namespace mpfem {
    class VariableNode;

    class ExpressionParser {
    public:
        ExpressionParser() = delete; // 纯静态工具类

        /**
         * @brief 将表达式字符串解析为纯 AST 树。
         * 变量被解析为 VariableRefNode，在 Manager::compile() 时链接。
         */
        static std::unique_ptr<VariableNode> parse(const std::string& expression);
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EXPRESSION_PARSER_HPP
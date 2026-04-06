#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include "core/tensor_shape.hpp"
#include "core/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mpfem {

    class ExpressionParser {
    public:
        class ExpressionProgram {
        public:
            struct Impl;
            ExpressionProgram();
            explicit ExpressionProgram(std::unique_ptr<Impl> impl) noexcept;
            ~ExpressionProgram();
            ExpressionProgram(ExpressionProgram&&) noexcept;
            ExpressionProgram& operator=(ExpressionProgram&&) noexcept;
            ExpressionProgram(const ExpressionProgram&) = delete;
            ExpressionProgram& operator=(const ExpressionProgram&) = delete;

            bool valid() const;
            TensorShape shape() const;
            const std::vector<std::string>& dependencies() const;
            ExprValue evaluate(std::span<const double> values) const;

        private:
            std::unique_ptr<Impl> impl_;
        };

        ExpressionParser(const ExpressionParser&) = delete;
        ExpressionParser& operator=(const ExpressionParser&) = delete;

        ExpressionParser();
        ~ExpressionParser();

        ExpressionProgram compile(const std::string& expression) const;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EXPRESSION_PARSER_HPP

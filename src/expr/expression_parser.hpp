#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include "core/types.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace mpfem {

    class ExpressionParser {
    public:
        struct VariableBinding {
            std::string name;
            double* ref = nullptr;
        };

        class ScalarProgram {
        public:
            struct Impl;
            ScalarProgram();
            explicit ScalarProgram(std::unique_ptr<Impl> impl) noexcept;
            ~ScalarProgram();
            ScalarProgram(ScalarProgram&&) noexcept;
            ScalarProgram& operator=(ScalarProgram&&) noexcept;
            ScalarProgram(const ScalarProgram&) = delete;
            ScalarProgram& operator=(const ScalarProgram&) = delete;

            bool valid() const;
            const std::vector<std::string>& dependencies() const;
            double evaluate(const std::unordered_map<std::string, double>& values = {}) const;

        private:
            std::unique_ptr<Impl> impl_;
        };

        class MatrixProgram {
        public:
            struct Impl;
            MatrixProgram();
            explicit MatrixProgram(std::unique_ptr<Impl> impl) noexcept;
            ~MatrixProgram();
            MatrixProgram(MatrixProgram&&) noexcept;
            MatrixProgram& operator=(MatrixProgram&&) noexcept;
            MatrixProgram(const MatrixProgram&) = delete;
            MatrixProgram& operator=(const MatrixProgram&) = delete;

            bool valid() const;
            const std::vector<std::string>& dependencies() const;
            Matrix3 evaluate(const std::unordered_map<std::string, double>& values = {}) const;

        private:
            std::unique_ptr<Impl> impl_;
        };

        ExpressionParser(const ExpressionParser&) = delete;
        ExpressionParser& operator=(const ExpressionParser&) = delete;

        ExpressionParser();
        ~ExpressionParser();

        ScalarProgram compileScalar(const std::string& expression,
            const std::vector<VariableBinding>& bindings) const;
        MatrixProgram compileMatrix(const std::string& expression,
            const std::vector<VariableBinding>& bindings) const;

        // Convenience one-shot APIs for configuration parsing.
        double evaluate(const std::string& expression,
            const std::map<std::string, double>& variables = {});
        Matrix3 evaluateMatrix(const std::string& expression,
            const std::map<std::string, double>& variables = {});
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EXPRESSION_PARSER_HPP

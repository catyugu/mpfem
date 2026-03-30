#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include "core/types.hpp"

#include <map>
#include <memory>
#include <string>
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
        ScalarProgram();
        ~ScalarProgram();
        ScalarProgram(ScalarProgram&&) noexcept;
        ScalarProgram& operator=(ScalarProgram&&) noexcept;
        ScalarProgram(const ScalarProgram&) = delete;
        ScalarProgram& operator=(const ScalarProgram&) = delete;

        [[nodiscard]] bool valid() const;
        [[nodiscard]] double evaluate() const;

    private:
        friend class ExpressionParser;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class MatrixProgram {
    public:
        MatrixProgram();
        ~MatrixProgram();
        MatrixProgram(MatrixProgram&&) noexcept;
        MatrixProgram& operator=(MatrixProgram&&) noexcept;
        MatrixProgram(const MatrixProgram&) = delete;
        MatrixProgram& operator=(const MatrixProgram&) = delete;

        [[nodiscard]] bool valid() const;
        [[nodiscard]] Matrix3 evaluate() const;

    private:
        friend class ExpressionParser;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    static ExpressionParser& instance();

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

}  // namespace mpfem

#endif  // MPFEM_EXPR_EXPRESSION_PARSER_HPP

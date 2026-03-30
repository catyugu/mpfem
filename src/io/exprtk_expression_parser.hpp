#ifndef MPFEM_EXPRTK_EXPRESSION_PARSER_HPP
#define MPFEM_EXPRTK_EXPRESSION_PARSER_HPP

#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace mpfem {

/**
 * @brief Expression parser using ExprTk with unit-aware evaluation.
 * 
 * Design:
 * - Variables bound by reference via pointer storage
 * - Unit handling: "110[GPa]" → 1.1e11
 * - Unified expression cache
 * - Thread-safe via thread-local singleton
 */
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

    ScalarProgram compileScalar(const std::string& expr,
                               const std::vector<VariableBinding>& bindings) const;
    MatrixProgram compileMatrix(const std::string& expr,
                               const std::vector<VariableBinding>& bindings) const;

    // Evaluate scalar expression
    // Unit handling: "110[GPa]" → 1.1e11
    double evaluate(const std::string& expr,
                   const std::map<std::string, double>& variables = {});

    // Evaluate matrix expression: {'a','b',...} format or scalar→diagonal
    Matrix3 evaluateMatrix(const std::string& expr,
                          const std::map<std::string, double>& variables = {});

    void clearCache();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mpfem

#endif

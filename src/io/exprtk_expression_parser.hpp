#ifndef MPFEM_EXPRTK_EXPRESSION_PARSER_HPP
#define MPFEM_EXPRTK_EXPRESSION_PARSER_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "io/unit_parser.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

// ExprTk header - single file library
#include <exprtk.hpp>

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
        ScalarProgram() = default;
        ScalarProgram(ScalarProgram&&) noexcept = default;
        ScalarProgram& operator=(ScalarProgram&&) noexcept = default;
        ScalarProgram(const ScalarProgram&) = delete;
        ScalarProgram& operator=(const ScalarProgram&) = delete;

        [[nodiscard]] bool valid() const { return expression_ != nullptr; }
        [[nodiscard]] double evaluate() const;

    private:
        friend class ExpressionParser;
        double multiplier_ = 1.0;
        std::unique_ptr<exprtk::symbol_table<double>> symbolTable_;
        std::unique_ptr<exprtk::expression<double>> expression_;
    };

    class MatrixProgram {
    public:
        MatrixProgram() = default;
        MatrixProgram(MatrixProgram&&) noexcept = default;
        MatrixProgram& operator=(MatrixProgram&&) noexcept = default;
        MatrixProgram(const MatrixProgram&) = delete;
        MatrixProgram& operator=(const MatrixProgram&) = delete;

        [[nodiscard]] Matrix3 evaluate() const;

    private:
        friend class ExpressionParser;
        bool literalMatrix_ = false;
        std::vector<ScalarProgram> components_;
    };

    static ExpressionParser& instance();
    ExpressionParser(const ExpressionParser&) = delete;
    ExpressionParser& operator=(const ExpressionParser&) = delete;

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
    ExpressionParser() = default;
    ~ExpressionParser() = default;

    struct CachedExpression {
        std::unique_ptr<exprtk::symbol_table<double>> symbolTable;
        std::unique_ptr<exprtk::expression<double>> expression;
        std::unique_ptr<exprtk::parser<double>> parser;
        std::vector<double> varStorage;
        std::vector<std::string> varNames;
        std::unordered_map<std::string, size_t> varIndex;
        bool compiled = false;
    };

    struct MatrixTemplate {
        bool literalMatrix = false;
        std::vector<std::string> components;
    };

    std::unordered_map<std::string, size_t> cacheIndex_;
    std::vector<CachedExpression> compiled_;
    std::unordered_map<std::string, MatrixTemplate> matrixTemplateCache_;

    double evaluateImpl(const std::string& exprStripped,
                       double unitMultiplier,
                       const std::map<std::string, double>& variables);

    static bool hasSameVariableSignature(const CachedExpression& cached,
                                        const std::map<std::string, double>& variables);
    const MatrixTemplate& matrixTemplateFor(const std::string& expr);
    static MatrixTemplate parseMatrixTemplate(std::string_view expr);
};

} // namespace mpfem

#endif

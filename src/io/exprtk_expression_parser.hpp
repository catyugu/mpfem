#ifndef MPFEM_EXPRTK_EXPRESSION_PARSER_HPP
#define MPFEM_EXPRTK_EXPRESSION_PARSER_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "io/unit_parser.hpp"
#include <string>
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
    static ExpressionParser& instance();
    ExpressionParser(const ExpressionParser&) = delete;
    ExpressionParser& operator=(const ExpressionParser&) = delete;

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

    // Use unique_ptr because exprtk types are not movable
    struct CachedExpression {
        std::unique_ptr<exprtk::expression<double>> expression;
        std::unique_ptr<exprtk::parser<double>> parser;
        std::vector<double> varStorage;
        std::vector<std::string> varNames;
        bool compiled = false;
    };

    std::unordered_map<std::string, size_t> cacheIndex_;
    std::vector<CachedExpression> compiled_;

    double evaluateImpl(const std::string& exprStripped,
                       double unitMultiplier,
                       const std::map<std::string, double>& variables);
};

} // namespace mpfem

#endif

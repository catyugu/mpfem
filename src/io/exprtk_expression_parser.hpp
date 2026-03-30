#ifndef MPFEM_EXPRTK_EXPRESSION_PARSER_HPP
#define MPFEM_EXPRTK_EXPRESSION_PARSER_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include <string>
#include <map>
#include <vector>

// ExprTk header - single file library
#include <exprtk.hpp>

namespace mpfem {

/**
 * @brief Expression parser using ExprTk with thread-safe caching.
 * 
 * Provides evaluation of scalar and matrix expressions with support for
 * variables T (temperature), V (voltage), and u (displacement vector).
 * 
 * Thread-safe by design through thread-local singleton instance.
 */
class ExpressionParser {
public:
    // Singleton access
    static ExpressionParser& instance();
    
    // Prevent copying
    ExpressionParser(const ExpressionParser&) = delete;
    ExpressionParser& operator=(const ExpressionParser&) = delete;

    /**
     * @brief Evaluate a scalar expression.
     * @param expr Expression string, e.g., "1.72e-8*(1+0.0039*(T-298))"
     * @param variables Map of variable names to values
     * @return Evaluated scalar result
     */
    double evaluateScalar(const std::string& expr,
                         const std::map<std::string, double>& variables = {});

    /**
     * @brief Evaluate a matrix expression to a 3x3 Matrix3.
     * @param expr Expression string for each component, or single scalar to create diagonal
     * @param variables Map of variable names to values
     * @return Evaluated Matrix3 result
     */
    Matrix3 evaluateMatrix(const std::string& expr,
                           const std::map<std::string, double>& variables = {});

    /**
     * @brief Clear the expression cache.
     */
    void clearCache();

private:
    // Private constructor for singleton
    ExpressionParser() = default;

    // Thread-safe cache entry - stores compiled expression and variable storage
    struct CachedExpression {
        exprtk::expression<double> expression;
        exprtk::parser<double> parser;
        std::vector<double> varStorage;  // Persistent storage for variables
        std::vector<std::string> varNames;
        bool compiled = false;
    };

    // Cache: expression string -> compiled expression
    using ExpressionCache = std::unordered_map<std::string, CachedExpression>;

    // Separate caches for scalar and matrix expressions
    ExpressionCache scalarCache_;
    ExpressionCache matrixCache_;

    // Helper to compile and evaluate scalar expression
    double compileAndEvaluate(const std::string& expr,
                             const std::map<std::string, double>& variables,
                             ExpressionCache& cache);

    // Parse matrix format {'a','b',...} with expression support
    Matrix3 parseMatrixWithExpressions(const std::string& expr,
                                      const std::map<std::string, double>& variables);
};

/**
 * @brief Check if value contains operators indicating it's an expression.
 */
inline bool isExpression(const std::string& value) {
    for (char c : value) {
        if (c == '(' || c == ')' || c == '*' || c == '/' || 
            c == '+' || c == '-' || c == '^') {
            return true;
        }
    }
    return false;
}

}  // namespace mpfem

#endif  // MPFEM_EXPRTK_EXPRESSION_PARSER_HPP

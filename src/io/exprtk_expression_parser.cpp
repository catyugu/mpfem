#include "io/exprtk_expression_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
#include <cctype>

namespace mpfem {

ExpressionParser& ExpressionParser::instance() {
    thread_local ExpressionParser parser;
    return parser;
}

double ExpressionParser::compileAndEvaluate(
    const std::string& expr,
    const std::map<std::string, double>& variables,
    ExpressionCache& cache) 
{
    // First check if it's a simple number
    if (!isExpression(expr)) {
        try {
            return std::stod(expr);
        } catch (...) {
            // Fall through to expression evaluation
        }
    }

    // Check cache
    auto it = cache.find(expr);
    if (it == cache.end()) {
        it = cache.try_emplace(expr).first;
    }
    
    CachedExpression& cached = it->second;
    
    // Rebuild variable list if needed
    std::vector<std::string> currentVarNames;
    for (const auto& [name, _] : variables) {
        currentVarNames.push_back(name);
    }
    
    // Recompile if variables changed or not yet compiled
    if (!cached.compiled || cached.varNames != currentVarNames) {
        cached.varNames = currentVarNames;
        cached.varStorage.resize(currentVarNames.size(), 0.0);
        
        // Rebuild symbol table
        exprtk::symbol_table<double> symbolTable;
        for (size_t i = 0; i < currentVarNames.size(); ++i) {
            symbolTable.add_variable(currentVarNames[i], cached.varStorage[i]);
        }
        symbolTable.add_constants();
        
        cached.expression.register_symbol_table(symbolTable);
        
        if (!cached.parser.compile(expr, cached.expression)) {
            MPFEM_THROW(ArgumentException,
                        "Expression compilation failed: " + expr +
                        " | error: " + cached.parser.error());
        }
        cached.compiled = true;
    }
    
    // Update variable values
    for (size_t i = 0; i < currentVarNames.size(); ++i) {
        auto varIt = variables.find(currentVarNames[i]);
        if (varIt == variables.end()) {
            MPFEM_THROW(ArgumentException,
                        "Missing variable '" + currentVarNames[i] +
                        "' for expression: " + expr);
        }
        cached.varStorage[i] = varIt->second;
    }
    
    return cached.expression.value();
}

double ExpressionParser::evaluateScalar(
    const std::string& expr,
    const std::map<std::string, double>& variables) 
{
    return compileAndEvaluate(expr, variables, scalarCache_);
}

Matrix3 ExpressionParser::evaluateMatrix(
    const std::string& expr,
    const std::map<std::string, double>& variables) 
{
    return parseMatrixWithExpressions(expr, variables);
}

Matrix3 ExpressionParser::parseMatrixWithExpressions(
    const std::string& expr,
    const std::map<std::string, double>& variables) 
{
    std::string trimmed = expr;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) {
        trimmed.erase(trimmed.begin());
    }
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) {
        trimmed.pop_back();
    }
    
    // Check for matrix format {'a','b',...}
    if (trimmed.size() >= 2 && trimmed[0] == '{') {
        std::vector<double> values;
        std::string token;
        bool inQuote = false;
        
        for (size_t i = 1; i < trimmed.size(); ++i) {
            char c = trimmed[i];
            if (c == '\'') {
                inQuote = !inQuote;
                if (!inQuote && !token.empty()) {
                    // Evaluate token as expression or number
                    if (isExpression(token)) {
                        values.push_back(evaluateScalar(token, variables));
                    } else {
                        values.push_back(std::stod(token));
                    }
                    token.clear();
                }
            } else if (inQuote) {
                token.push_back(c);
            }
        }
        
        if (values.size() == 9) {
            Matrix3 m;
            m << values[0], values[3], values[6],
                 values[1], values[4], values[7],
                 values[2], values[5], values[8];
            return m;
        } else if (values.size() == 1) {
            return Matrix3::Identity() * values[0];
        } else {
            MPFEM_THROW(ArgumentException,
                        "Invalid matrix expression component count: " + expr);
        }
    }
    
    // Not a matrix format - evaluate as scalar and create diagonal matrix
    double scalar = evaluateScalar(expr, variables);
    return Matrix3::Identity() * scalar;
}

void ExpressionParser::clearCache() {
    scalarCache_.clear();
    matrixCache_.clear();
}

}  // namespace mpfem

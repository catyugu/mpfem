#include "io/exprtk_expression_parser.hpp"
#include "core/exception.hpp"
#include <cctype>

namespace mpfem
{

    ExpressionParser &ExpressionParser::instance()
    {
        thread_local ExpressionParser parser;
        return parser;
    }

    double ExpressionParser::evaluate(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        // Find unit bracket [ ... ]
        size_t start = expr.find('[');
        size_t end = expr.find(']', start);

        std::string exprStr = expr;
        double multiplier = 1.0;

        if (start != std::string::npos && end != std::string::npos && end > start)
        {
            std::string unit = expr.substr(start + 1, end - start - 1);
            exprStr = expr.substr(0, start);

            // Parse unit multiplier
            if (unit == "GPa")
                multiplier = 1e9;
            else if (unit == "MPa")
                multiplier = 1e6;
            else if (unit == "kPa")
                multiplier = 1e3;
            else if (unit == "Pa")
                multiplier = 1.0;
            else if (unit == "W/(m*K)" || unit == "W m^-1 K^-1")
                multiplier = 1.0;
            else if (unit == "J/(kg*K)")
                multiplier = 1.0;
            else if (unit == "J")
                multiplier = 1.0;
            else if (unit == "kJ")
                multiplier = 1e3;
            else if (unit == "W")
                multiplier = 1.0;
            else if (unit == "K")
                multiplier = 1.0;
            else if (unit == "kg")
                multiplier = 1.0;
            else if (unit == "g")
                multiplier = 1e-3;
            else if (unit == "m")
                multiplier = 1.0;
            else if (unit == "cm")
                multiplier = 1e-2;
            else if (unit == "mm")
                multiplier = 1e-3;
            else if (unit == "s")
                multiplier = 1.0;
            else if (unit == "S/m")
                multiplier = 1.0;
            else if (unit == "1/K")
                multiplier = 1.0;
            else if (unit == "1/m")
                multiplier = 1.0;
            // kg/m^3, etc. - compound units have multiplier 1.0

            // Trim whitespace from expression
            while (!exprStr.empty() && std::isspace(static_cast<unsigned char>(exprStr.back())))
            {
                exprStr.pop_back();
            }
        }

        return evaluateImpl(exprStr, multiplier, variables);
    }

    double ExpressionParser::evaluateImpl(
        const std::string &exprStripped,
        double unitMultiplier,
        const std::map<std::string, double> &variables)
    {
        // Check cache
        auto it = cacheIndex_.find(exprStripped);
        if (it == cacheIndex_.end())
        {
            size_t newIndex = compiled_.size();
            cacheIndex_[exprStripped] = newIndex;
            compiled_.resize(newIndex + 1);
            compiled_[newIndex].expression = std::make_unique<exprtk::expression<double>>();
            compiled_[newIndex].parser = std::make_unique<exprtk::parser<double>>();
            it = cacheIndex_.find(exprStripped);
        }

        CachedExpression &cached = compiled_[it->second];

        // Collect variable names
        std::vector<std::string> varNames;
        for (const auto &[name, _] : variables)
        {
            varNames.push_back(name);
        }

        // Recompile if needed
        if (!cached.compiled || cached.varNames != varNames)
        {
            cached.varNames = varNames;
            cached.varStorage.resize(varNames.size(), 0.0);

            exprtk::symbol_table<double> symbolTable;
            for (size_t i = 0; i < varNames.size(); ++i)
            {
                symbolTable.add_variable(varNames[i], cached.varStorage[i]);
            }
            symbolTable.add_constants();

            cached.expression->register_symbol_table(symbolTable);

            if (!cached.parser->compile(exprStripped, *cached.expression))
            {
                MPFEM_THROW(ArgumentException,
                            "Expression compilation failed: " + exprStripped +
                                " | error: " + cached.parser->error());
            }
            cached.compiled = true;
        }

        // Update variable values
        for (size_t i = 0; i < varNames.size(); ++i)
        {
            auto varIt = variables.find(varNames[i]);
            if (varIt != variables.end())
            {
                cached.varStorage[i] = varIt->second;
            }
        }

        return cached.expression->value() * unitMultiplier;
    }

    Matrix3 ExpressionParser::evaluateMatrix(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        std::string trimmed = expr;
        while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())))
        {
            trimmed.erase(trimmed.begin());
        }
        while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())))
        {
            trimmed.pop_back();
        }

        // Matrix format {'a','b',...}
        if (trimmed.size() >= 2 && trimmed[0] == '{')
        {
            std::vector<double> values;
            std::string token;
            bool inQuote = false;

            for (size_t i = 1; i < trimmed.size(); ++i)
            {
                char c = trimmed[i];
                if (c == '\'')
                {
                    inQuote = !inQuote;
                    if (!inQuote && !token.empty())
                    {
                        values.push_back(evaluate(token, variables));
                        token.clear();
                    }
                }
                else if (inQuote)
                {
                    token.push_back(c);
                }
            }

            if (values.size() == 9)
            {
                Matrix3 m;
                m << values[0], values[3], values[6],
                    values[1], values[4], values[7],
                    values[2], values[5], values[8];
                return m;
            }
            else if (values.size() == 1)
            {
                return Matrix3::Identity() * values[0];
            }
            else
            {
                MPFEM_THROW(ArgumentException,
                            "Invalid matrix expression: expected 9 components, got " +
                                std::to_string(values.size()));
            }
        }

        // Scalar → diagonal matrix
        double scalar = evaluate(expr, variables);
        return Matrix3::Identity() * scalar;
    }

    void ExpressionParser::clearCache()
    {
        cacheIndex_.clear();
        compiled_.clear();
    }

} // namespace mpfem

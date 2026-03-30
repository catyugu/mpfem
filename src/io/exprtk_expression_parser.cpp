#include "io/exprtk_expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>

namespace mpfem
{

    namespace
    {

        bool isSeparator(const char c)
        {
            return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
        }

    } // namespace

    double ExpressionParser::ScalarProgram::evaluate() const
    {
        MPFEM_ASSERT(expression_ != nullptr, "Attempting to evaluate an invalid scalar expression program.");
        return expression_->value() * multiplier_;
    }

    Matrix3 ExpressionParser::MatrixProgram::evaluate() const
    {
        MPFEM_ASSERT(!components_.empty(), "Attempting to evaluate an empty matrix expression program.");

        if (!literalMatrix_)
        {
            const double scalar = components_.front().evaluate();
            return Matrix3::Identity() * scalar;
        }

        if (components_.size() == 1)
        {
            const double scalar = components_.front().evaluate();
            return Matrix3::Identity() * scalar;
        }

        MPFEM_ASSERT(components_.size() == 9,
                     "Invalid matrix expression program: expected 1 or 9 components.");

        Matrix3 matrix;
        matrix << components_[0].evaluate(), components_[3].evaluate(), components_[6].evaluate(),
            components_[1].evaluate(), components_[4].evaluate(), components_[7].evaluate(),
            components_[2].evaluate(), components_[5].evaluate(), components_[8].evaluate();
        return matrix;
    }

    ExpressionParser &ExpressionParser::instance()
    {
        thread_local ExpressionParser parser;
        return parser;
    }

    ExpressionParser::ScalarProgram ExpressionParser::compileScalar(
        const std::string &expr,
        const std::vector<VariableBinding> &bindings) const
    {
        auto unitResult = UnitRegistry::instance().stripUnit(expr);
        std::string expressionText = strings::trim(std::string(unitResult.expression));
        if (expressionText.empty())
        {
            MPFEM_THROW(ArgumentException, "Expression is empty after unit stripping: " + expr);
        }

        ScalarProgram program;
        program.multiplier_ = unitResult.multiplier;
        program.symbolTable_ = std::make_unique<exprtk::symbol_table<double>>();
        program.expression_ = std::make_unique<exprtk::expression<double>>();

        for (const auto &binding : bindings)
        {
            if (binding.ref == nullptr)
            {
                MPFEM_THROW(ArgumentException, "Variable binding has null reference for variable: " + binding.name);
            }
            if (!program.symbolTable_->add_variable(binding.name, *binding.ref))
            {
                MPFEM_THROW(ArgumentException, "Failed to bind variable for expression compilation: " + binding.name);
            }
        }
        program.symbolTable_->add_constants();
        program.expression_->register_symbol_table(*program.symbolTable_);

        exprtk::parser<double> parser;
        if (!parser.compile(expressionText, *program.expression_))
        {
            MPFEM_THROW(ArgumentException,
                        "Expression compilation failed: " + expressionText +
                            " | error: " + parser.error());
        }

        return program;
    }

    ExpressionParser::MatrixProgram ExpressionParser::compileMatrix(
        const std::string &expr,
        const std::vector<VariableBinding> &bindings) const
    {
        MatrixTemplate tmpl = parseMatrixTemplate(expr);

        MatrixProgram program;
        program.literalMatrix_ = tmpl.literalMatrix;

        if (!tmpl.literalMatrix)
        {
            program.components_.push_back(compileScalar(expr, bindings));
            return program;
        }

        program.components_.reserve(tmpl.components.size());
        for (const std::string &componentExpr : tmpl.components)
        {
            program.components_.push_back(compileScalar(componentExpr, bindings));
        }
        return program;
    }

    double ExpressionParser::evaluate(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        // Use UnitRegistry for unit parsing - two phase
        auto unitResult = UnitRegistry::instance().stripUnit(expr);
        std::string exprStr = strings::trim(std::string(unitResult.expression));
        double multiplier = unitResult.multiplier;
        // No more manual bracket finding or if-else chain
        return evaluateImpl(exprStr, multiplier, variables);
    }

    double ExpressionParser::evaluateImpl(
        const std::string &exprStripped,
        double unitMultiplier,
        const std::map<std::string, double> &variables)
    {
        auto it = cacheIndex_.find(exprStripped);
        if (it == cacheIndex_.end())
        {
            const size_t newIndex = compiled_.size();
            cacheIndex_[exprStripped] = newIndex;
            compiled_.emplace_back();
            compiled_[newIndex].expression = std::make_unique<exprtk::expression<double>>();
            compiled_[newIndex].parser = std::make_unique<exprtk::parser<double>>();
            it = cacheIndex_.find(exprStripped);
        }

        CachedExpression &cached = compiled_[it->second];

        if (!cached.compiled || !hasSameVariableSignature(cached, variables))
        {
            cached.varNames.clear();
            cached.varNames.reserve(variables.size());
            cached.varStorage.assign(variables.size(), 0.0);
            cached.varIndex.clear();
            cached.varIndex.reserve(variables.size());

            cached.symbolTable = std::make_unique<exprtk::symbol_table<double>>();

            size_t index = 0;
            for (const auto &[name, value] : variables)
            {
                cached.varNames.push_back(name);
                cached.varStorage[index] = value;
                cached.varIndex[name] = index;
                if (!cached.symbolTable->add_variable(name, cached.varStorage[index]))
                {
                    MPFEM_THROW(ArgumentException,
                                "Failed to bind variable for expression compilation: " + name);
                }
                ++index;
            }
            cached.symbolTable->add_constants();

            cached.expression = std::make_unique<exprtk::expression<double>>();
            cached.expression->register_symbol_table(*cached.symbolTable);

            cached.parser = std::make_unique<exprtk::parser<double>>();

            if (!cached.parser->compile(exprStripped, *cached.expression))
            {
                MPFEM_THROW(ArgumentException,
                            "Expression compilation failed: " + exprStripped +
                                " | error: " + cached.parser->error());
            }
            cached.compiled = true;
        }

        for (const auto &[name, value] : variables)
        {
            auto indexIt = cached.varIndex.find(name);
            if (indexIt != cached.varIndex.end())
            {
                cached.varStorage[indexIt->second] = value;
            }
        }

        return cached.expression->value() * unitMultiplier;
    }

    Matrix3 ExpressionParser::evaluateMatrix(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        const MatrixTemplate &tmpl = matrixTemplateFor(expr);
        if (!tmpl.literalMatrix)
        {
            const double scalar = evaluate(expr, variables);
            return Matrix3::Identity() * scalar;
        }

        if (tmpl.components.size() == 1)
        {
            const double scalar = evaluate(tmpl.components.front(), variables);
            return Matrix3::Identity() * scalar;
        }

        MPFEM_ASSERT(tmpl.components.size() == 9,
                     "Invalid matrix expression template: expected 1 or 9 components.");

        std::array<double, 9> values{};
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = evaluate(tmpl.components[i], variables);
        }

        Matrix3 matrix;
        matrix << values[0], values[3], values[6],
            values[1], values[4], values[7],
            values[2], values[5], values[8];
        return matrix;
    }

    void ExpressionParser::clearCache()
    {
        cacheIndex_.clear();
        compiled_.clear();
        matrixTemplateCache_.clear();
    }

    bool ExpressionParser::hasSameVariableSignature(
        const CachedExpression &cached,
        const std::map<std::string, double> &variables)
    {
        if (cached.varNames.size() != variables.size())
        {
            return false;
        }

        auto variableIt = variables.begin();
        for (const std::string &cachedName : cached.varNames)
        {
            if (variableIt == variables.end() || cachedName != variableIt->first)
            {
                return false;
            }
            ++variableIt;
        }
        return true;
    }

    const ExpressionParser::MatrixTemplate &ExpressionParser::matrixTemplateFor(const std::string &expr)
    {
        auto cacheIt = matrixTemplateCache_.find(expr);
        if (cacheIt != matrixTemplateCache_.end())
        {
            return cacheIt->second;
        }

        auto [it, _] = matrixTemplateCache_.emplace(expr, parseMatrixTemplate(expr));
        return it->second;
    }

    ExpressionParser::MatrixTemplate ExpressionParser::parseMatrixTemplate(std::string_view expr)
    {
        MatrixTemplate tmpl;
        const std::string trimmed = strings::trim(std::string(expr));
        if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}')
        {
            return tmpl;
        }

        tmpl.literalMatrix = true;
        const std::string_view content(trimmed.data() + 1, trimmed.size() - 2);

        size_t index = 0;
        while (index < content.size())
        {
            while (index < content.size() && isSeparator(content[index]))
            {
                ++index;
            }
            if (index >= content.size())
            {
                break;
            }
            if (content[index] != '\'')
            {
                MPFEM_THROW(ArgumentException,
                            "Invalid matrix expression literal. Expected quoted component near: " + trimmed);
            }

            const size_t endQuote = content.find('\'', index + 1);
            if (endQuote == std::string_view::npos)
            {
                MPFEM_THROW(ArgumentException,
                            "Invalid matrix expression literal. Unterminated quoted component: " + trimmed);
            }

            std::string component = strings::trim(std::string(content.substr(index + 1, endQuote - index - 1)));
            if (component.empty())
            {
                MPFEM_THROW(ArgumentException,
                            "Invalid matrix expression literal. Empty component in: " + trimmed);
            }
            tmpl.components.push_back(std::move(component));
            index = endQuote + 1;
        }

        if (tmpl.components.size() != 1 && tmpl.components.size() != 9)
        {
            MPFEM_THROW(ArgumentException,
                        "Invalid matrix expression: expected 1 or 9 components, got " +
                            std::to_string(tmpl.components.size()));
        }

        return tmpl;
    }

} // namespace mpfem

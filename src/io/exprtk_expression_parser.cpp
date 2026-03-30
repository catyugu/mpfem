#include "io/exprtk_expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "io/unit_parser.hpp"
#include <exprtk.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>
#include <unordered_map>

namespace mpfem
{

    namespace
    {

        bool isSeparator(const char c)
        {
            return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
        }

    } // namespace

    struct ExpressionParser::ScalarProgram::Impl
    {
        double multiplier = 1.0;
        std::unique_ptr<exprtk::symbol_table<double>> symbolTable;
        std::unique_ptr<exprtk::expression<double>> expression;
    };

    struct ExpressionParser::MatrixProgram::Impl
    {
        bool literalMatrix = false;
        std::vector<ExpressionParser::ScalarProgram> components;
    };

    struct ExpressionParser::Impl
    {
        struct CachedExpression
        {
            std::unique_ptr<exprtk::symbol_table<double>> symbolTable;
            std::unique_ptr<exprtk::expression<double>> expression;
            std::unique_ptr<exprtk::parser<double>> parser;
            std::vector<double> varStorage;
            std::vector<std::string> varNames;
            std::unordered_map<std::string, size_t> varIndex;
            bool compiled = false;
        };

        struct MatrixTemplate
        {
            bool literalMatrix = false;
            std::vector<std::string> components;
        };

        std::unordered_map<std::string, size_t> cacheIndex;
        std::vector<CachedExpression> compiled;
        std::unordered_map<std::string, MatrixTemplate> matrixTemplateCache;

        static bool hasSameVariableSignature(
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

        static MatrixTemplate parseMatrixTemplate(std::string_view expr)
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

        const MatrixTemplate &matrixTemplateFor(const std::string &expr)
        {
            auto cacheIt = matrixTemplateCache.find(expr);
            if (cacheIt != matrixTemplateCache.end())
            {
                return cacheIt->second;
            }

            auto [it, _] = matrixTemplateCache.emplace(expr, parseMatrixTemplate(expr));
            return it->second;
        }

        double evaluateImpl(
            const std::string &exprStripped,
            double unitMultiplier,
            const std::map<std::string, double> &variables)
        {
            auto it = cacheIndex.find(exprStripped);
            if (it == cacheIndex.end())
            {
                const size_t newIndex = compiled.size();
                cacheIndex[exprStripped] = newIndex;
                compiled.emplace_back();
                compiled[newIndex].expression = std::make_unique<exprtk::expression<double>>();
                compiled[newIndex].parser = std::make_unique<exprtk::parser<double>>();
                it = cacheIndex.find(exprStripped);
            }

            CachedExpression &cached = compiled[it->second];

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
    };

    ExpressionParser::ScalarProgram::ScalarProgram()
        : impl_(std::make_unique<Impl>())
    {
    }

    ExpressionParser::ScalarProgram::~ScalarProgram() = default;
    ExpressionParser::ScalarProgram::ScalarProgram(ScalarProgram &&) noexcept = default;
    ExpressionParser::ScalarProgram &ExpressionParser::ScalarProgram::operator=(ScalarProgram &&) noexcept = default;

    bool ExpressionParser::ScalarProgram::valid() const
    {
        return impl_ && impl_->expression != nullptr;
    }

    double ExpressionParser::ScalarProgram::evaluate() const
    {
        MPFEM_ASSERT(valid(), "Attempting to evaluate an invalid scalar expression program.");
        return impl_->expression->value() * impl_->multiplier;
    }

    ExpressionParser::MatrixProgram::MatrixProgram()
        : impl_(std::make_unique<Impl>())
    {
    }

    ExpressionParser::MatrixProgram::~MatrixProgram() = default;
    ExpressionParser::MatrixProgram::MatrixProgram(MatrixProgram &&) noexcept = default;
    ExpressionParser::MatrixProgram &ExpressionParser::MatrixProgram::operator=(MatrixProgram &&) noexcept = default;

    bool ExpressionParser::MatrixProgram::valid() const
    {
        return impl_ && !impl_->components.empty();
    }

    Matrix3 ExpressionParser::MatrixProgram::evaluate() const
    {
        MPFEM_ASSERT(valid(), "Attempting to evaluate an empty matrix expression program.");

        if (!impl_->literalMatrix)
        {
            const double scalar = impl_->components.front().evaluate();
            return Matrix3::Identity() * scalar;
        }

        if (impl_->components.size() == 1)
        {
            const double scalar = impl_->components.front().evaluate();
            return Matrix3::Identity() * scalar;
        }

        MPFEM_ASSERT(impl_->components.size() == 9,
                     "Invalid matrix expression program: expected 1 or 9 components.");

        Matrix3 matrix;
        matrix << impl_->components[0].evaluate(), impl_->components[3].evaluate(), impl_->components[6].evaluate(),
            impl_->components[1].evaluate(), impl_->components[4].evaluate(), impl_->components[7].evaluate(),
            impl_->components[2].evaluate(), impl_->components[5].evaluate(), impl_->components[8].evaluate();
        return matrix;
    }

    ExpressionParser::ExpressionParser()
        : impl_(std::make_unique<Impl>())
    {
    }

    ExpressionParser::~ExpressionParser() = default;

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
        program.impl_->multiplier = unitResult.multiplier;
        program.impl_->symbolTable = std::make_unique<exprtk::symbol_table<double>>();
        program.impl_->expression = std::make_unique<exprtk::expression<double>>();

        for (const auto &binding : bindings)
        {
            if (binding.ref == nullptr)
            {
                MPFEM_THROW(ArgumentException, "Variable binding has null reference for variable: " + binding.name);
            }
            if (!program.impl_->symbolTable->add_variable(binding.name, *binding.ref))
            {
                MPFEM_THROW(ArgumentException, "Failed to bind variable for expression compilation: " + binding.name);
            }
        }
        program.impl_->symbolTable->add_constants();
        program.impl_->expression->register_symbol_table(*program.impl_->symbolTable);

        exprtk::parser<double> parser;
        if (!parser.compile(expressionText, *program.impl_->expression))
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
        Impl::MatrixTemplate tmpl = Impl::parseMatrixTemplate(expr);

        MatrixProgram program;
        program.impl_->literalMatrix = tmpl.literalMatrix;

        if (!tmpl.literalMatrix)
        {
            program.impl_->components.push_back(compileScalar(expr, bindings));
            return program;
        }

        program.impl_->components.reserve(tmpl.components.size());
        for (const std::string &componentExpr : tmpl.components)
        {
            program.impl_->components.push_back(compileScalar(componentExpr, bindings));
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
        return impl_->evaluateImpl(exprStr, multiplier, variables);
    }

    Matrix3 ExpressionParser::evaluateMatrix(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        const Impl::MatrixTemplate &tmpl = impl_->matrixTemplateFor(expr);
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
        impl_->cacheIndex.clear();
        impl_->compiled.clear();
        impl_->matrixTemplateCache.clear();
    }

} // namespace mpfem

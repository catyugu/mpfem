#include "io/exprtk_expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "io/unit_parser.hpp"
#include <exprtk.hpp>
#include <array>
#include <cctype>
#include <string_view>
#include <utility>

namespace mpfem
{

    namespace
    {

        struct MatrixTemplate
        {
            bool literalMatrix = false;
            std::vector<std::string> components;
        };

        using VariableStorage = std::vector<std::pair<std::string, double>>;

        VariableStorage copyVariables(const std::map<std::string, double> &variables)
        {
            VariableStorage storage;
            storage.reserve(variables.size());
            for (const auto &[name, value] : variables)
            {
                storage.emplace_back(name, value);
            }
            return storage;
        }

        std::vector<ExpressionParser::VariableBinding> makeBindings(VariableStorage &storage)
        {
            std::vector<ExpressionParser::VariableBinding> bindings;
            bindings.reserve(storage.size());
            for (auto &[name, value] : storage)
            {
                bindings.push_back(ExpressionParser::VariableBinding{name, &value});
            }
            return bindings;
        }

        bool isSeparator(const char c)
        {
            return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
        }

        MatrixTemplate parseMatrixTemplate(std::string_view expr)
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

    ExpressionParser::ExpressionParser() = default;

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
        MatrixTemplate tmpl = parseMatrixTemplate(expr);

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
        VariableStorage storage = copyVariables(variables);
        auto bindings = makeBindings(storage);
        ScalarProgram program = compileScalar(expr, bindings);
        return program.evaluate();
    }

    Matrix3 ExpressionParser::evaluateMatrix(
        const std::string &expr,
        const std::map<std::string, double> &variables)
    {
        VariableStorage storage = copyVariables(variables);
        auto bindings = makeBindings(storage);
        MatrixProgram program = compileMatrix(expr, bindings);
        return program.evaluate();
    }

} // namespace mpfem

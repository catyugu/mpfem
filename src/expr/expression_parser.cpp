#include "expr/expression_parser.hpp"

#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <array>
#include <cctype>
#include <exprtk.hpp>
#include <string_view>
#include <utility>

namespace mpfem {
namespace {

struct MatrixTemplate {
    bool literalMatrix = false;
    std::vector<std::string> components;
};

using VariableStorage = std::vector<std::pair<std::string, double>>;

VariableStorage copyVariables(const std::map<std::string, double>& variables) {
    VariableStorage storage;
    storage.reserve(variables.size());
    for (const auto& [name, value] : variables) {
        storage.emplace_back(name, value);
    }
    return storage;
}

std::vector<ExpressionParser::VariableBinding> makeBindings(VariableStorage& storage) {
    std::vector<ExpressionParser::VariableBinding> bindings;
    bindings.reserve(storage.size());
    for (auto& [name, value] : storage) {
        bindings.push_back(ExpressionParser::VariableBinding{name, &value});
    }
    return bindings;
}

bool isSeparator(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
}

MatrixTemplate parseMatrixTemplate(std::string_view expression) {
    MatrixTemplate matrixTemplate;
    const std::string trimmed = strings::trim(std::string(expression));

    if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}') {
        return matrixTemplate;
    }

    matrixTemplate.literalMatrix = true;
    const std::string_view content(trimmed.data() + 1, trimmed.size() - 2);

    size_t index = 0;
    while (index < content.size()) {
        while (index < content.size() && isSeparator(content[index])) {
            ++index;
        }
        if (index >= content.size()) {
            break;
        }

        if (content[index] != '\'') {
            MPFEM_THROW(ArgumentException,
                        "Invalid matrix expression literal. Expected quoted component near: " + trimmed);
        }

        const size_t endQuote = content.find('\'', index + 1);
        if (endQuote == std::string_view::npos) {
            MPFEM_THROW(ArgumentException,
                        "Invalid matrix expression literal. Unterminated quoted component: " + trimmed);
        }

        std::string component = strings::trim(std::string(content.substr(index + 1, endQuote - index - 1)));
        if (component.empty()) {
            MPFEM_THROW(ArgumentException,
                        "Invalid matrix expression literal. Empty component in: " + trimmed);
        }

        matrixTemplate.components.push_back(std::move(component));
        index = endQuote + 1;
    }

    if (matrixTemplate.components.size() != 1 && matrixTemplate.components.size() != 9) {
        MPFEM_THROW(ArgumentException,
                    "Invalid matrix expression: expected 1 or 9 components, got " +
                        std::to_string(matrixTemplate.components.size()));
    }

    return matrixTemplate;
}

}  // namespace

struct ExpressionParser::ScalarProgram::Impl {
    double multiplier = 1.0;
    std::unique_ptr<exprtk::symbol_table<double>> symbolTable;
    std::unique_ptr<exprtk::expression<double>> expression;
};

struct ExpressionParser::MatrixProgram::Impl {
    bool literalMatrix = false;
    std::vector<ExpressionParser::ScalarProgram> components;
};

ExpressionParser::ScalarProgram::ScalarProgram()
    : impl_(std::make_unique<Impl>()) {
}

ExpressionParser::ScalarProgram::ScalarProgram(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ExpressionParser::ScalarProgram::~ScalarProgram() = default;
ExpressionParser::ScalarProgram::ScalarProgram(ScalarProgram&&) noexcept = default;
ExpressionParser::ScalarProgram& ExpressionParser::ScalarProgram::operator=(ScalarProgram&&) noexcept = default;

bool ExpressionParser::ScalarProgram::valid() const {
    return impl_ && impl_->expression != nullptr;
}

double ExpressionParser::ScalarProgram::evaluate() const {
    MPFEM_ASSERT(valid(), "Attempting to evaluate an invalid scalar expression program.");
    return impl_->expression->value() * impl_->multiplier;
}

ExpressionParser::MatrixProgram::MatrixProgram()
    : impl_(std::make_unique<Impl>()) {
}

ExpressionParser::MatrixProgram::MatrixProgram(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ExpressionParser::MatrixProgram::~MatrixProgram() = default;
ExpressionParser::MatrixProgram::MatrixProgram(MatrixProgram&&) noexcept = default;
ExpressionParser::MatrixProgram& ExpressionParser::MatrixProgram::operator=(MatrixProgram&&) noexcept = default;

bool ExpressionParser::MatrixProgram::valid() const {
    return impl_ && !impl_->components.empty();
}

Matrix3 ExpressionParser::MatrixProgram::evaluate() const {
    MPFEM_ASSERT(valid(), "Attempting to evaluate an empty matrix expression program.");

    if (!impl_->literalMatrix || impl_->components.size() == 1) {
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

ExpressionParser& ExpressionParser::instance() {
    thread_local ExpressionParser parser;
    return parser;
}

ExpressionParser::ScalarProgram ExpressionParser::compileScalar(
    const std::string& expression,
    const std::vector<VariableBinding>& bindings) const {
    const UnitParseResult unitResult = UnitRegistry::instance().stripUnit(expression);
    std::string expressionText = strings::trim(std::string(unitResult.expression));
    if (expressionText.empty()) {
        MPFEM_THROW(ArgumentException, "Expression is empty after unit stripping: " + expression);
    }

    auto impl = std::make_unique<ScalarProgram::Impl>();
    impl->multiplier = unitResult.multiplier;
    impl->symbolTable = std::make_unique<exprtk::symbol_table<double>>();
    impl->expression = std::make_unique<exprtk::expression<double>>();

    for (const auto& binding : bindings) {
        if (binding.ref == nullptr) {
            MPFEM_THROW(ArgumentException, "Variable binding has null reference for variable: " + binding.name);
        }
        if (!impl->symbolTable->add_variable(binding.name, *binding.ref)) {
            MPFEM_THROW(ArgumentException, "Failed to bind variable for expression compilation: " + binding.name);
        }
    }

    impl->symbolTable->add_constants();
    impl->expression->register_symbol_table(*impl->symbolTable);

    exprtk::parser<double> parser;
    if (!parser.compile(expressionText, *impl->expression)) {
        MPFEM_THROW(ArgumentException,
                    "Expression compilation failed: " + expressionText + " | error: " + parser.error());
    }

    return ScalarProgram(std::move(impl));
}

ExpressionParser::MatrixProgram ExpressionParser::compileMatrix(
    const std::string& expression,
    const std::vector<VariableBinding>& bindings) const {
    MatrixTemplate matrixTemplate = parseMatrixTemplate(expression);

    auto impl = std::make_unique<MatrixProgram::Impl>();
    impl->literalMatrix = matrixTemplate.literalMatrix;

    if (!matrixTemplate.literalMatrix) {
        impl->components.push_back(compileScalar(expression, bindings));
        return MatrixProgram(std::move(impl));
    }

    impl->components.reserve(matrixTemplate.components.size());
    for (const std::string& componentExpression : matrixTemplate.components) {
        impl->components.push_back(compileScalar(componentExpression, bindings));
    }
    return MatrixProgram(std::move(impl));
}

double ExpressionParser::evaluate(
    const std::string& expression,
    const std::map<std::string, double>& variables) {
    VariableStorage storage = copyVariables(variables);
    auto bindings = makeBindings(storage);
    ScalarProgram program = compileScalar(expression, bindings);
    return program.evaluate();
}

Matrix3 ExpressionParser::evaluateMatrix(
    const std::string& expression,
    const std::map<std::string, double>& variables) {
    VariableStorage storage = copyVariables(variables);
    auto bindings = makeBindings(storage);
    MatrixProgram program = compileMatrix(expression, bindings);
    return program.evaluate();
}

}  // namespace mpfem

#include "expr/expression_parser.hpp"

#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_set>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace mpfem {
namespace {

struct MatrixTemplate {
    bool literalMatrix = false;
    std::vector<std::string> components;
};

struct AstNode {
    enum class Kind {
        Constant,
        Variable,
        Add,
        Subtract,
        Multiply,
        Divide,
        Power,
        Negate,
        Sin,
        Cos,
        Tan,
        Exp,
        Log,
        Sqrt,
        Abs,
        Min,
        Max,
    };

    Kind kind = Kind::Constant;
    double value = 0.0;
    std::string variableName;
    std::unique_ptr<AstNode> lhs;
    std::unique_ptr<AstNode> rhs;
};

void collectDependencies(const AstNode& node, std::unordered_set<std::string>& seen, std::vector<std::string>& deps)
{
    if (node.kind == AstNode::Kind::Variable) {
        if (seen.insert(node.variableName).second) {
            deps.push_back(node.variableName);
        }
        return;
    }

    if (node.lhs) {
        collectDependencies(*node.lhs, seen, deps);
    }
    if (node.rhs) {
        collectDependencies(*node.rhs, seen, deps);
    }
}

bool isSeparator(char c)
{
    return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
}

MatrixTemplate parseMatrixTemplate(std::string_view expression)
{
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

class ScalarAstCompiler {
public:
    explicit ScalarAstCompiler(std::string_view text)
        : text_(text)
    {
    }

    std::unique_ptr<AstNode> compile()
    {
        auto root = parseExpression();
        skipWhitespace();
        if (!eof()) {
            MPFEM_THROW(ArgumentException,
                        "Unexpected token near: " + std::string(text_.substr(pos_)));
        }
        return root;
    }

private:
    std::unique_ptr<AstNode> parseExpression()
    {
        auto lhs = parseTerm();
        for (;;) {
            skipWhitespace();
            if (match('+')) {
                lhs = makeBinary(AstNode::Kind::Add, std::move(lhs), parseTerm());
                continue;
            }
            if (match('-')) {
                lhs = makeBinary(AstNode::Kind::Subtract, std::move(lhs), parseTerm());
                continue;
            }
            return lhs;
        }
    }

    std::unique_ptr<AstNode> parseTerm()
    {
        auto lhs = parsePower();
        for (;;) {
            skipWhitespace();
            if (match('*')) {
                lhs = makeBinary(AstNode::Kind::Multiply, std::move(lhs), parsePower());
                continue;
            }
            if (match('/')) {
                lhs = makeBinary(AstNode::Kind::Divide, std::move(lhs), parsePower());
                continue;
            }
            return lhs;
        }
    }

    std::unique_ptr<AstNode> parsePower()
    {
        auto lhs = parseUnary();
        skipWhitespace();
        if (!match('^')) {
            return lhs;
        }
        return makeBinary(AstNode::Kind::Power, std::move(lhs), parsePower());
    }

    std::unique_ptr<AstNode> parseUnary()
    {
        skipWhitespace();
        if (match('+')) {
            return parseUnary();
        }
        if (match('-')) {
            auto node = std::make_unique<AstNode>();
            node->kind = AstNode::Kind::Negate;
            node->lhs = parseUnary();
            return node;
        }
        return parsePrimary();
    }

    std::unique_ptr<AstNode> parsePrimary()
    {
        skipWhitespace();
        if (match('(')) {
            auto inner = parseExpression();
            skipWhitespace();
            if (!match(')')) {
                MPFEM_THROW(ArgumentException, "Missing closing ')' in expression.");
            }
            return inner;
        }

        if (peekIsNumberStart()) {
            return parseNumber();
        }

        if (peekIsIdentifierStart()) {
            const std::string name = parseIdentifier();
            skipWhitespace();
            if (match('(')) {
                return parseFunction(name);
            }
            return parseVariableOrConstant(name);
        }

        MPFEM_THROW(ArgumentException,
                    "Unexpected token near: " + std::string(text_.substr(pos_)));
    }

    std::unique_ptr<AstNode> parseFunction(const std::string& name)
    {
        auto arg0 = parseExpression();
        skipWhitespace();

        if (match(')')) {
            AstNode::Kind kind;
            if (name == "sin") kind = AstNode::Kind::Sin;
            else if (name == "cos") kind = AstNode::Kind::Cos;
            else if (name == "tan") kind = AstNode::Kind::Tan;
            else if (name == "exp") kind = AstNode::Kind::Exp;
            else if (name == "log") kind = AstNode::Kind::Log;
            else if (name == "sqrt") kind = AstNode::Kind::Sqrt;
            else if (name == "abs") kind = AstNode::Kind::Abs;
            else {
                MPFEM_THROW(ArgumentException,
                            "Unsupported unary function: " + name);
            }
            auto node = std::make_unique<AstNode>();
            node->kind = kind;
            node->lhs = std::move(arg0);
            return node;
        }

        if (!match(',')) {
            MPFEM_THROW(ArgumentException,
                        "Expected ',' or ')' in function call: " + name);
        }

        auto arg1 = parseExpression();
        skipWhitespace();
        if (!match(')')) {
            MPFEM_THROW(ArgumentException,
                        "Expected ')' in function call: " + name);
        }

        if (name == "pow") {
            return makeBinary(AstNode::Kind::Power, std::move(arg0), std::move(arg1));
        }
        if (name == "min") {
            return makeBinary(AstNode::Kind::Min, std::move(arg0), std::move(arg1));
        }
        if (name == "max") {
            return makeBinary(AstNode::Kind::Max, std::move(arg0), std::move(arg1));
        }

        MPFEM_THROW(ArgumentException,
                    "Unsupported binary function: " + name);
    }

    std::unique_ptr<AstNode> parseVariableOrConstant(const std::string& name)
    {
        auto node = std::make_unique<AstNode>();

        if (name == "pi") {
            node->kind = AstNode::Kind::Constant;
            node->value = 3.141592653589793238462643383279502884;
            return node;
        }
        if (name == "e") {
            node->kind = AstNode::Kind::Constant;
            node->value = 2.718281828459045235360287471352662498;
            return node;
        }

        node->kind = AstNode::Kind::Variable;
        node->variableName = name;
        return node;
    }

    std::unique_ptr<AstNode> parseNumber()
    {
        const char* begin = text_.data() + pos_;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);

        if (end == begin) {
            MPFEM_THROW(ArgumentException,
                        "Invalid numeric literal near: " + std::string(text_.substr(pos_)));
        }

        pos_ += static_cast<size_t>(end - begin);
        auto node = std::make_unique<AstNode>();
        node->kind = AstNode::Kind::Constant;
        node->value = value;
        return node;
    }

    std::string parseIdentifier()
    {
        const size_t begin = pos_;
        ++pos_;
        while (!eof()) {
            const char c = text_[pos_];
            if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_') {
                break;
            }
            ++pos_;
        }
        return std::string(text_.substr(begin, pos_ - begin));
    }

    std::unique_ptr<AstNode> makeBinary(AstNode::Kind kind,
                                        std::unique_ptr<AstNode> lhs,
                                        std::unique_ptr<AstNode> rhs)
    {
        auto node = std::make_unique<AstNode>();
        node->kind = kind;
        node->lhs = std::move(lhs);
        node->rhs = std::move(rhs);
        return node;
    }

    bool peekIsIdentifierStart() const
    {
        if (eof()) {
            return false;
        }
        const char c = text_[pos_];
        return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
    }

    bool peekIsNumberStart() const
    {
        if (eof()) {
            return false;
        }
        const char c = text_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.') {
            return true;
        }
        if ((c == '+' || c == '-') && pos_ + 1 < text_.size()) {
            const char next = text_[pos_ + 1];
            return std::isdigit(static_cast<unsigned char>(next)) != 0 || next == '.';
        }
        return false;
    }

    void skipWhitespace()
    {
        while (!eof() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
    }

    bool match(char expected)
    {
        if (eof() || text_[pos_] != expected) {
            return false;
        }
        ++pos_;
        return true;
    }

    bool eof() const { return pos_ >= text_.size(); }

    std::string_view text_;
    size_t pos_ = 0;
};

double evalAstNode(const AstNode& node, const std::unordered_map<std::string, double>& values){
    switch (node.kind) {
        case AstNode::Kind::Constant:
            return node.value;
        case AstNode::Kind::Variable:
        {
            const auto it = values.find(node.variableName);
            if (it == values.end()) {
                MPFEM_THROW(ArgumentException, "Unbound variable in expression: " + node.variableName);
            }
            return it->second;
        }
        case AstNode::Kind::Add:
            return evalAstNode(*node.lhs, values) + evalAstNode(*node.rhs, values);
        case AstNode::Kind::Subtract:
            return evalAstNode(*node.lhs, values) - evalAstNode(*node.rhs, values);
        case AstNode::Kind::Multiply:
            return evalAstNode(*node.lhs, values) * evalAstNode(*node.rhs, values);
        case AstNode::Kind::Divide:
            return evalAstNode(*node.lhs, values) / evalAstNode(*node.rhs, values);
        case AstNode::Kind::Power:
            return std::pow(evalAstNode(*node.lhs, values), evalAstNode(*node.rhs, values));
        case AstNode::Kind::Negate:
            return -evalAstNode(*node.lhs, values);
        case AstNode::Kind::Sin:
            return std::sin(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Cos:
            return std::cos(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Tan:
            return std::tan(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Exp:
            return std::exp(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Log:
            return std::log(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Sqrt:
            return std::sqrt(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Abs:
            return std::abs(evalAstNode(*node.lhs, values));
        case AstNode::Kind::Min:
            return std::min(evalAstNode(*node.lhs, values), evalAstNode(*node.rhs, values));
        case AstNode::Kind::Max:
            return std::max(evalAstNode(*node.lhs, values), evalAstNode(*node.rhs, values));
    }

    MPFEM_THROW(ArgumentException, "Unknown AST node kind.");
}

}  // namespace

struct ExpressionParser::ScalarProgram::Impl {
    double multiplier = 1.0;
    std::unique_ptr<AstNode> root;
    std::vector<std::string> dependencies;
};

struct ExpressionParser::MatrixProgram::Impl {
    bool literalMatrix = false;
    std::vector<ExpressionParser::ScalarProgram> components;
    std::vector<std::string> dependencies;
};

ExpressionParser::ScalarProgram::ScalarProgram()
    : impl_(std::make_unique<Impl>())
{
}

ExpressionParser::ScalarProgram::ScalarProgram(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

ExpressionParser::ScalarProgram::~ScalarProgram() = default;
ExpressionParser::ScalarProgram::ScalarProgram(ScalarProgram&&) noexcept = default;
ExpressionParser::ScalarProgram& ExpressionParser::ScalarProgram::operator=(ScalarProgram&&) noexcept = default;

bool ExpressionParser::ScalarProgram::valid() const
{
    return impl_ && impl_->root != nullptr;
}

const std::vector<std::string>& ExpressionParser::ScalarProgram::dependencies() const
{
    static const std::vector<std::string> empty;
    return impl_ ? impl_->dependencies : empty;
}

double ExpressionParser::ScalarProgram::evaluate(const std::unordered_map<std::string, double>& values) const
{
    MPFEM_ASSERT(valid(), "Attempting to evaluate an invalid scalar expression program.");
    return evalAstNode(*impl_->root, values) * impl_->multiplier;
}

ExpressionParser::MatrixProgram::MatrixProgram()
    : impl_(std::make_unique<Impl>())
{
}

ExpressionParser::MatrixProgram::MatrixProgram(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

ExpressionParser::MatrixProgram::~MatrixProgram() = default;
ExpressionParser::MatrixProgram::MatrixProgram(MatrixProgram&&) noexcept = default;
ExpressionParser::MatrixProgram& ExpressionParser::MatrixProgram::operator=(MatrixProgram&&) noexcept = default;

bool ExpressionParser::MatrixProgram::valid() const
{
    return impl_ && !impl_->components.empty();
}

const std::vector<std::string>& ExpressionParser::MatrixProgram::dependencies() const
{
    static const std::vector<std::string> empty;
    return impl_ ? impl_->dependencies : empty;
}

Matrix3 ExpressionParser::MatrixProgram::evaluate(const std::unordered_map<std::string, double>& values) const
{
    MPFEM_ASSERT(valid(), "Attempting to evaluate an empty matrix expression program.");

    if (!impl_->literalMatrix || impl_->components.size() == 1) {
        const double scalar = impl_->components.front().evaluate(values);
        return Matrix3::Identity() * scalar;
    }

    MPFEM_ASSERT(impl_->components.size() == 9,
                 "Invalid matrix expression program: expected 1 or 9 components.");

    Matrix3 matrix;
    matrix << impl_->components[0].evaluate(values), impl_->components[3].evaluate(values), impl_->components[6].evaluate(values),
        impl_->components[1].evaluate(values), impl_->components[4].evaluate(values), impl_->components[7].evaluate(values),
        impl_->components[2].evaluate(values), impl_->components[5].evaluate(values), impl_->components[8].evaluate(values);
    return matrix;
}

ExpressionParser::ExpressionParser() = default;
ExpressionParser::~ExpressionParser() = default;

ExpressionParser::ScalarProgram ExpressionParser::compileScalar(const std::string& expression) const
{
    UnitRegistry registry;
    const UnitParseResult unitResult = registry.stripUnit(expression);
    const std::string expressionText = strings::trim(std::string(unitResult.expression));
    if (expressionText.empty()) {
        MPFEM_THROW(ArgumentException, "Expression is empty after unit stripping: " + expression);
    }

    ScalarAstCompiler compiler(expressionText);

    auto impl = std::make_unique<ScalarProgram::Impl>();
    impl->multiplier = unitResult.multiplier;
    impl->root = compiler.compile();
    std::unordered_set<std::string> seen;
    collectDependencies(*impl->root, seen, impl->dependencies);
    return ScalarProgram(std::move(impl));
}

ExpressionParser::MatrixProgram ExpressionParser::compileMatrix(const std::string& expression) const
{
    MatrixTemplate matrixTemplate = parseMatrixTemplate(expression);

    auto impl = std::make_unique<MatrixProgram::Impl>();
    impl->literalMatrix = matrixTemplate.literalMatrix;

    if (!matrixTemplate.literalMatrix) {
        impl->components.push_back(compileScalar(expression));
        impl->dependencies = impl->components.front().dependencies();
        return MatrixProgram(std::move(impl));
    }

    impl->components.reserve(matrixTemplate.components.size());
    for (const std::string& componentExpression : matrixTemplate.components) {
        impl->components.push_back(compileScalar(componentExpression));
        for (const std::string& dep : impl->components.back().dependencies()) {
            if (std::find(impl->dependencies.begin(), impl->dependencies.end(), dep) == impl->dependencies.end()) {
                impl->dependencies.push_back(dep);
            }
        }
    }
    return MatrixProgram(std::move(impl));
}

}  // namespace mpfem

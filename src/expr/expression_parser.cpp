#include "expr/expression_parser.hpp"

#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <unordered_set>

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
                VectorLiteral,
                MatrixLiteral,
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
                Dot,
                Sym,
                Trace,
                Transpose,
            };

            Kind kind = Kind::Constant;
            Real value = 0.0;
            std::string variableName;
            std::vector<std::unique_ptr<AstNode>> args;
        };

        bool isSeparator(char c)
        {
            return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',';
        }

        MatrixTemplate parseComsolMatrixTemplate(std::string_view expression)
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
                        "Invalid comsol matrix literal. Expected quoted component near: " + trimmed);
                }

                const size_t endQuote = content.find('\'', index + 1);
                if (endQuote == std::string_view::npos) {
                    MPFEM_THROW(ArgumentException,
                        "Invalid comsol matrix literal. Unterminated quoted component: " + trimmed);
                }

                std::string component = strings::trim(std::string(content.substr(index + 1, endQuote - index - 1)));
                if (component.empty()) {
                    MPFEM_THROW(ArgumentException, "Invalid comsol matrix literal with empty component.");
                }

                matrixTemplate.components.push_back(std::move(component));
                index = endQuote + 1;
            }

            if (matrixTemplate.components.size() != 1 && matrixTemplate.components.size() != 9) {
                MPFEM_THROW(ArgumentException,
                    "Invalid comsol matrix literal: expected 1 or 9 components, got " + std::to_string(matrixTemplate.components.size()));
            }

            return matrixTemplate;
        }

        std::string convertComsolMatrixToBracketLiteral(const MatrixTemplate& tpl)
        {
            MPFEM_ASSERT(tpl.literalMatrix, "Matrix template must be comsol literal.");
            if (tpl.components.size() == 1) {
                const std::string& c = tpl.components[0];
                return "[" + c + ",0,0;0," + c + ",0;0,0," + c + "]";
            }

            MPFEM_ASSERT(tpl.components.size() == 9, "Comsol matrix literal expects 9 components.");
            return "["
                + tpl.components[0] + "," + tpl.components[3] + "," + tpl.components[6] + ";"
                + tpl.components[1] + "," + tpl.components[4] + "," + tpl.components[7] + ";"
                + tpl.components[2] + "," + tpl.components[5] + "," + tpl.components[8] + "]";
        }

        bool isNearOne(Real value)
        {
            return std::abs(value - 1.0) < 1e-15;
        }

        class TensorAstCompiler {
        public:
            explicit TensorAstCompiler(std::string_view text)
                : text_(text)
            {
            }

            std::unique_ptr<AstNode> compile()
            {
                auto root = parseExpression();
                skipWhitespace();
                if (!eof()) {
                    MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));
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
                    node->args.push_back(parseUnary());
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
                    return applyUnitSuffixIfPresent(std::move(inner));
                }

                if (peek() == '[') {
                    return applyUnitSuffixIfPresent(parseBracketLiteral());
                }

                if (peekIsNumberStart()) {
                    return applyUnitSuffixIfPresent(parseNumber());
                }

                if (peekIsIdentifierStart()) {
                    const std::string name = parseIdentifier();
                    skipWhitespace();
                    if (match('(')) {
                        return applyUnitSuffixIfPresent(parseFunction(name));
                    }
                    return applyUnitSuffixIfPresent(parseVariableOrConstant(name));
                }

                MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));
            }

            std::unique_ptr<AstNode> parseBracketLiteral()
            {
                if (!match('[')) {
                    MPFEM_THROW(ArgumentException, "Internal parser error: bracket literal expected.");
                }

                std::vector<std::vector<std::unique_ptr<AstNode>>> rows;
                rows.emplace_back();

                for (;;) {
                    skipWhitespace();
                    const size_t partBegin = pos_;
                    int parenDepth = 0;
                    int unitBracketDepth = 0;
                    while (!eof()) {
                        const char c = text_[pos_];
                        if (c == '(') {
                            ++parenDepth;
                        }
                        else if (c == ')') {
                            --parenDepth;
                        }
                        else if (c == '[') {
                            ++unitBracketDepth;
                        }
                        else if (c == ']') {
                            if (unitBracketDepth > 0) {
                                --unitBracketDepth;
                            }
                            else if (parenDepth == 0) {
                                break;
                            }
                        }
                        else if (parenDepth == 0 && unitBracketDepth == 0 && (c == ',' || c == ';')) {
                            break;
                        }
                        ++pos_;
                    }

                    if (eof()) {
                        MPFEM_THROW(ArgumentException, "Missing closing ']' in literal expression.");
                    }

                    const std::string part = strings::trim(std::string(text_.substr(partBegin, pos_ - partBegin)));
                    if (part.empty()) {
                        MPFEM_THROW(ArgumentException, "Empty component in bracket literal.");
                    }

                    UnitRegistry unitRegistry;
                    const UnitParseResult unitPart = unitRegistry.stripUnit(part);
                    TensorAstCompiler scalarCompiler(unitPart.expression);
                    auto component = scalarCompiler.compile();
                    rows.back().push_back(applyMultiplierIfNeeded(std::move(component), unitPart.multiplier));

                    const char delim = text_[pos_];
                    ++pos_;
                    if (delim == ',') {
                        continue;
                    }
                    if (delim == ';') {
                        rows.emplace_back();
                        continue;
                    }
                    if (delim == ']') {
                        break;
                    }
                }

                skipWhitespace();
                if (match('^')) {
                    skipWhitespace();
                    if (!match('T')) {
                        MPFEM_THROW(ArgumentException, "Expected '^T' after vector literal.");
                    }
                }

                const size_t rowCount = rows.size();
                const size_t colCount = rows.front().size();
                for (const auto& row : rows) {
                    if (row.size() != colCount) {
                        MPFEM_THROW(ArgumentException, "Inconsistent row size in bracket literal.");
                    }
                }

                if (rowCount == 1 && colCount == 1) {
                    return std::move(rows.front().front());
                }

                auto node = std::make_unique<AstNode>();
                if (rowCount == 1 && colCount == 3) {
                    node->kind = AstNode::Kind::VectorLiteral;
                    for (auto& entry : rows.front()) {
                        node->args.push_back(std::move(entry));
                    }
                    return node;
                }

                if (rowCount == 3 && colCount == 3) {
                    node->kind = AstNode::Kind::MatrixLiteral;
                    for (auto& row : rows) {
                        for (auto& entry : row) {
                            node->args.push_back(std::move(entry));
                        }
                    }
                    return node;
                }

                MPFEM_THROW(ArgumentException,
                    "Unsupported bracket literal shape [" + std::to_string(rowCount) + "x" + std::to_string(colCount) + "]");
            }

            std::unique_ptr<AstNode> parseFunction(const std::string& name)
            {
                std::vector<std::unique_ptr<AstNode>> args;
                skipWhitespace();
                if (!match(')')) {
                    for (;;) {
                        args.push_back(parseExpression());
                        skipWhitespace();
                        if (match(')')) {
                            break;
                        }
                        if (!match(',')) {
                            MPFEM_THROW(ArgumentException,
                                "Expected ',' or ')' in function call: " + name);
                        }
                    }
                }

                auto node = std::make_unique<AstNode>();
                if (name == "dot") {
                    if (args.size() != 2) {
                        MPFEM_THROW(ArgumentException, "dot(a,b) requires exactly 2 arguments.");
                    }
                    node->kind = AstNode::Kind::Dot;
                    node->args = std::move(args);
                    return node;
                }

                if (name == "sym" || name == "trace" || name == "tr" || name == "transpose") {
                    if (args.size() != 1) {
                        MPFEM_THROW(ArgumentException, name + "() requires exactly 1 argument.");
                    }
                    if (name == "sym") {
                        node->kind = AstNode::Kind::Sym;
                    }
                    else if (name == "transpose") {
                        node->kind = AstNode::Kind::Transpose;
                    }
                    else {
                        node->kind = AstNode::Kind::Trace;
                    }
                    node->args = std::move(args);
                    return node;
                }

                if (args.size() == 1) {
                    if (name == "grad") {
                        // grad(V) is treated as a variable reference to "grad(V)"
                        // Shape inference happens at DAG layer via registeredShapes
                        if (args[0]->kind != AstNode::Kind::Variable) {
                            MPFEM_THROW(ArgumentException, "grad() requires a single field variable argument.");
                        }
                        auto gradVar = std::make_unique<AstNode>();
                        gradVar->kind = AstNode::Kind::Variable;
                        gradVar->variableName = "grad(" + args[0]->variableName + ")";
                        return gradVar;
                    }
                    if (name == "sin")
                        node->kind = AstNode::Kind::Sin;
                    else if (name == "cos")
                        node->kind = AstNode::Kind::Cos;
                    else if (name == "tan")
                        node->kind = AstNode::Kind::Tan;
                    else if (name == "exp")
                        node->kind = AstNode::Kind::Exp;
                    else if (name == "log")
                        node->kind = AstNode::Kind::Log;
                    else if (name == "sqrt")
                        node->kind = AstNode::Kind::Sqrt;
                    else if (name == "abs")
                        node->kind = AstNode::Kind::Abs;
                    else {
                        MPFEM_THROW(ArgumentException, "Unsupported unary function: " + name);
                    }
                    node->args = std::move(args);
                    return node;
                }

                if (args.size() == 2) {
                    if (name == "pow")
                        node->kind = AstNode::Kind::Power;
                    else if (name == "min")
                        node->kind = AstNode::Kind::Min;
                    else if (name == "max")
                        node->kind = AstNode::Kind::Max;
                    else {
                        MPFEM_THROW(ArgumentException, "Unsupported binary function: " + name);
                    }
                    node->args = std::move(args);
                    return node;
                }

                MPFEM_THROW(ArgumentException, "Unsupported arity for function: " + name);
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
                const Real value = std::strtod(begin, &end);
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
                node->args.push_back(std::move(lhs));
                node->args.push_back(std::move(rhs));
                return node;
            }

            static std::unique_ptr<AstNode> applyMultiplierIfNeeded(std::unique_ptr<AstNode> node, Real multiplier)
            {
                if (isNearOne(multiplier)) {
                    return node;
                }
                auto mulNode = std::make_unique<AstNode>();
                mulNode->kind = AstNode::Kind::Multiply;
                auto c = std::make_unique<AstNode>();
                c->kind = AstNode::Kind::Constant;
                c->value = multiplier;
                mulNode->args.push_back(std::move(c));
                mulNode->args.push_back(std::move(node));
                return mulNode;
            }

            std::unique_ptr<AstNode> applyUnitSuffixIfPresent(std::unique_ptr<AstNode> node)
            {
                skipWhitespace();
                if (!match('[')) {
                    return node;
                }

                const size_t unitBegin = pos_;
                while (!eof() && text_[pos_] != ']') {
                    ++pos_;
                }
                if (eof()) {
                    MPFEM_THROW(ArgumentException, "Missing closing ']' for unit suffix.");
                }
                const std::string unit = strings::trim(std::string(text_.substr(unitBegin, pos_ - unitBegin)));
                ++pos_; // consume ']'

                UnitRegistry registry;
                const Real multiplier = registry.getMultiplier(unit);
                return applyMultiplierIfNeeded(std::move(node), multiplier);
            }

            char peek() const
            {
                return eof() ? '\0' : text_[pos_];
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

        void collectDependencies(const AstNode& node, std::unordered_set<std::string>& seen, std::vector<std::string>& deps)
        {
            if (node.kind == AstNode::Kind::Variable) {
                if (seen.insert(node.variableName).second) {
                    deps.push_back(node.variableName);
                }
                return;
            }
            for (const auto& arg : node.args) {
                collectDependencies(*arg, seen, deps);
            }
        }

        TensorShape inferShape(const AstNode& node,
            const std::unordered_map<std::string, TensorShape>& registeredShapes)
        {
            switch (node.kind) {
            case AstNode::Kind::Constant:
                return TensorShape::scalar();
            case AstNode::Kind::Variable:
                if (const auto it = registeredShapes.find(node.variableName); it != registeredShapes.end()) {
                    return it->second;
                }
                return TensorShape::scalar();
            case AstNode::Kind::VectorLiteral:
                return TensorShape::vector(3);
            case AstNode::Kind::MatrixLiteral:
                return TensorShape::matrix(3, 3);
            case AstNode::Kind::Dot:
                return TensorShape::scalar();
            case AstNode::Kind::Trace:
                return TensorShape::scalar();
            case AstNode::Kind::Sym:
            case AstNode::Kind::Transpose:
                return inferShape(*node.args[0], registeredShapes);
            case AstNode::Kind::Add:
            case AstNode::Kind::Subtract:
                return inferShape(*node.args[0], registeredShapes);
            case AstNode::Kind::Multiply: {
                const TensorShape lhs = inferShape(*node.args[0], registeredShapes);
                const TensorShape rhs = inferShape(*node.args[1], registeredShapes);
                if (lhs.isScalar())
                    return rhs;
                if (rhs.isScalar())
                    return lhs;
                if (lhs.isMatrix() && rhs.isVector())
                    return TensorShape::vector(3);
                if (lhs.isMatrix() && rhs.isMatrix())
                    return TensorShape::matrix(3, 3);
                MPFEM_THROW(ArgumentException, "Unsupported multiply shape combination.");
            }
            case AstNode::Kind::Divide:
                return inferShape(*node.args[0], registeredShapes);
            case AstNode::Kind::Power:
            case AstNode::Kind::Negate:
            case AstNode::Kind::Sin:
            case AstNode::Kind::Cos:
            case AstNode::Kind::Tan:
            case AstNode::Kind::Exp:
            case AstNode::Kind::Log:
            case AstNode::Kind::Sqrt:
            case AstNode::Kind::Abs:
            case AstNode::Kind::Min:
            case AstNode::Kind::Max:
                return inferShape(*node.args[0], registeredShapes);
            }
            MPFEM_THROW(ArgumentException, "Unknown AST node kind for shape inference.");
        }

        TensorValue evaluateAst(const AstNode& node,
            std::span<const TensorValue> vars,
            const std::unordered_map<std::string, size_t>& dependencyIndices)
        {
            switch (node.kind) {
            case AstNode::Kind::Constant:
                return TensorValue::scalar(node.value);
            case AstNode::Kind::Variable: {
                const auto it = dependencyIndices.find(node.variableName);
                if (it == dependencyIndices.end() || it->second >= vars.size()) {
                    MPFEM_THROW(ArgumentException, "Variable index binding failed: " + node.variableName);
                }
                return vars[it->second];
            }
            case AstNode::Kind::VectorLiteral:
                return TensorValue::vector(
                    evaluateAst(*node.args[0], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[1], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[2], vars, dependencyIndices).asScalar());
            case AstNode::Kind::MatrixLiteral:
                return TensorValue::matrix3(
                    evaluateAst(*node.args[0], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[1], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[2], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[3], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[4], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[5], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[6], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[7], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[8], vars, dependencyIndices).asScalar());
            case AstNode::Kind::Add:
                return evaluateAst(*node.args[0], vars, dependencyIndices) + evaluateAst(*node.args[1], vars, dependencyIndices);
            case AstNode::Kind::Subtract:
                return evaluateAst(*node.args[0], vars, dependencyIndices) - evaluateAst(*node.args[1], vars, dependencyIndices);
            case AstNode::Kind::Multiply:
                return evaluateAst(*node.args[0], vars, dependencyIndices) * evaluateAst(*node.args[1], vars, dependencyIndices);
            case AstNode::Kind::Divide:
                return evaluateAst(*node.args[0], vars, dependencyIndices) / evaluateAst(*node.args[1], vars, dependencyIndices);
            case AstNode::Kind::Power:
                return TensorValue::scalar(std::pow(
                    evaluateAst(*node.args[0], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[1], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Negate:
                return -evaluateAst(*node.args[0], vars, dependencyIndices);
            case AstNode::Kind::Sin:
                return TensorValue::scalar(std::sin(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Cos:
                return TensorValue::scalar(std::cos(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Tan:
                return TensorValue::scalar(std::tan(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Exp:
                return TensorValue::scalar(std::exp(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Log:
                return TensorValue::scalar(std::log(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Sqrt:
                return TensorValue::scalar(std::sqrt(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Abs:
                return TensorValue::scalar(std::abs(evaluateAst(*node.args[0], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Min:
                return TensorValue::scalar(std::min(
                    evaluateAst(*node.args[0], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[1], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Max:
                return TensorValue::scalar(std::max(
                    evaluateAst(*node.args[0], vars, dependencyIndices).asScalar(),
                    evaluateAst(*node.args[1], vars, dependencyIndices).asScalar()));
            case AstNode::Kind::Dot:
                return TensorValue::scalar(dot(
                    evaluateAst(*node.args[0], vars, dependencyIndices),
                    evaluateAst(*node.args[1], vars, dependencyIndices)));
            case AstNode::Kind::Sym:
                return sym(evaluateAst(*node.args[0], vars, dependencyIndices));
            case AstNode::Kind::Trace:
                return TensorValue::scalar(trace(evaluateAst(*node.args[0], vars, dependencyIndices)));
            case AstNode::Kind::Transpose:
                return transpose(evaluateAst(*node.args[0], vars, dependencyIndices));
            }
            MPFEM_THROW(ArgumentException, "Unknown AST node kind in evaluation.");
        }

    } // namespace

    struct ExpressionParser::ExpressionProgram::Impl {
        TensorShape shape = TensorShape::scalar();
        std::unique_ptr<AstNode> root;
        std::vector<std::string> dependencies;
        std::unordered_map<std::string, size_t> dependencyIndices;
    };

    ExpressionParser::ExpressionProgram::ExpressionProgram()
        : impl_(std::make_unique<Impl>())
    {
    }

    ExpressionParser::ExpressionProgram::ExpressionProgram(std::unique_ptr<Impl> impl) noexcept
        : impl_(std::move(impl))
    {
    }

    ExpressionParser::ExpressionProgram::~ExpressionProgram() = default;
    ExpressionParser::ExpressionProgram::ExpressionProgram(ExpressionProgram&&) noexcept = default;
    ExpressionParser::ExpressionProgram& ExpressionParser::ExpressionProgram::operator=(ExpressionProgram&&) noexcept = default;

    bool ExpressionParser::ExpressionProgram::valid() const
    {
        return impl_ && impl_->root != nullptr;
    }

    TensorShape ExpressionParser::ExpressionProgram::shape() const
    {
        return impl_ ? impl_->shape : TensorShape::scalar();
    }

    const std::vector<std::string>& ExpressionParser::ExpressionProgram::dependencies() const
    {
        static const std::vector<std::string> empty;
        return impl_ ? impl_->dependencies : empty;
    }

    TensorValue ExpressionParser::ExpressionProgram::evaluate(std::span<const TensorValue> values) const
    {
        MPFEM_ASSERT(valid(), "Attempting to evaluate an invalid expression program.");
        MPFEM_ASSERT(values.size() == impl_->dependencies.size(),
            "Expression input size does not match dependency size.");

        return evaluateAst(*impl_->root, values, impl_->dependencyIndices);
    }

    ExpressionParser::ExpressionParser() = default;
    ExpressionParser::~ExpressionParser() = default;

    ExpressionParser::ExpressionProgram ExpressionParser::compile(const std::string& expression,
        const std::unordered_map<std::string, TensorShape>& registeredShapes) const
    {
        std::string expressionText = strings::trim(expression);

        MatrixTemplate comsol = parseComsolMatrixTemplate(expressionText);
        if (comsol.literalMatrix) {
            expressionText = convertComsolMatrixToBracketLiteral(comsol);
        }

        if (expressionText.empty()) {
            MPFEM_THROW(ArgumentException, "Expression is empty after unit stripping: " + expression);
        }

        auto impl = std::make_unique<ExpressionProgram::Impl>();

        TensorAstCompiler compiler(expressionText);
        auto root = compiler.compile();
        impl->shape = inferShape(*root, registeredShapes);

        std::unordered_set<std::string> seen;
        collectDependencies(*root, seen, impl->dependencies);

        impl->dependencyIndices.clear();
        impl->dependencyIndices.reserve(impl->dependencies.size());
        for (size_t i = 0; i < impl->dependencies.size(); ++i) {
            impl->dependencyIndices.emplace(impl->dependencies[i], i);
        }

        impl->root = std::move(root);

        return ExpressionProgram(std::move(impl));
    }

    ExpressionParser::ExpressionProgram ExpressionParser::compile(const std::string& expression) const
    {
        static const std::unordered_map<std::string, TensorShape> emptyShapes;
        return compile(expression, emptyShapes);
    }

} // namespace mpfem

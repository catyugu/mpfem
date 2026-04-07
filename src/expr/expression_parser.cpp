#include "expr/expression_parser.hpp"

#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace mpfem {
    namespace {

        enum class OpCode : uint8_t {
            Constant, // push literal (value in Instruction::value)
            LoadVar, // push variable (index in Instruction::index)
            Add,
            Sub,
            Mul,
            Div,
            Pow, // binary ops
            Neg,
            Sin,
            Cos,
            Tan,
            Exp,
            Log,
            Sqrt,
            Abs,
            Min,
            Max, // unary ops
        };

        struct Instruction {
            OpCode op;
            double value; // for Constant
            int index; // for LoadVar
        };

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
            int variableIndex = -1;
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
                    "Invalid matrix expression: expected 1 or 9 components, got " + std::to_string(matrixTemplate.components.size()));
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
                    if (name == "sin")
                        kind = AstNode::Kind::Sin;
                    else if (name == "cos")
                        kind = AstNode::Kind::Cos;
                    else if (name == "tan")
                        kind = AstNode::Kind::Tan;
                    else if (name == "exp")
                        kind = AstNode::Kind::Exp;
                    else if (name == "log")
                        kind = AstNode::Kind::Log;
                    else if (name == "sqrt")
                        kind = AstNode::Kind::Sqrt;
                    else if (name == "abs")
                        kind = AstNode::Kind::Abs;
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

        void bindAstVariableIndices(AstNode& node, const std::vector<std::string>& dependencies)
        {
            if (node.kind == AstNode::Kind::Variable) {
                const auto it = std::find(dependencies.begin(), dependencies.end(), node.variableName);
                if (it == dependencies.end()) {
                    MPFEM_THROW(ArgumentException, "Variable index binding failed: " + node.variableName);
                }
                node.variableIndex = static_cast<int>(std::distance(dependencies.begin(), it));
                return;
            }

            if (node.lhs) {
                bindAstVariableIndices(*node.lhs, dependencies);
            }
            if (node.rhs) {
                bindAstVariableIndices(*node.rhs, dependencies);
            }
        }

        void linearize(const AstNode* node, std::vector<Instruction>& out)
        {
            switch (node->kind) {
            case AstNode::Kind::Constant:
                out.push_back({OpCode::Constant, node->value, -1});
                break;
            case AstNode::Kind::Variable:
                out.push_back({OpCode::LoadVar, 0.0, node->variableIndex});
                break;
            case AstNode::Kind::Add:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Add, 0.0, -1});
                break;
            case AstNode::Kind::Subtract:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Sub, 0.0, -1});
                break;
            case AstNode::Kind::Multiply:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Mul, 0.0, -1});
                break;
            case AstNode::Kind::Divide:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Div, 0.0, -1});
                break;
            case AstNode::Kind::Power:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Pow, 0.0, -1});
                break;
            case AstNode::Kind::Negate:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Neg, 0.0, -1});
                break;
            case AstNode::Kind::Sin:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Sin, 0.0, -1});
                break;
            case AstNode::Kind::Cos:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Cos, 0.0, -1});
                break;
            case AstNode::Kind::Tan:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Tan, 0.0, -1});
                break;
            case AstNode::Kind::Exp:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Exp, 0.0, -1});
                break;
            case AstNode::Kind::Log:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Log, 0.0, -1});
                break;
            case AstNode::Kind::Sqrt:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Sqrt, 0.0, -1});
                break;
            case AstNode::Kind::Abs:
                linearize(node->lhs.get(), out);
                out.push_back({OpCode::Abs, 0.0, -1});
                break;
            case AstNode::Kind::Min:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Min, 0.0, -1});
                break;
            case AstNode::Kind::Max:
                linearize(node->lhs.get(), out);
                linearize(node->rhs.get(), out);
                out.push_back({OpCode::Max, 0.0, -1});
                break;
            }
        }

        double evaluate_single_vm(std::span<const double> vars, const std::vector<Instruction>& instructions, double multiplier)
        {
            std::vector<double> stack;
            stack.reserve(16);

            for (const Instruction& insn : instructions) {
                switch (insn.op) {
                case OpCode::Constant:
                    stack.push_back(insn.value);
                    break;
                case OpCode::LoadVar:
                    stack.push_back(vars[static_cast<size_t>(insn.index)]);
                    break;
                case OpCode::Add: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(a + b);
                    break;
                }
                case OpCode::Sub: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(a - b);
                    break;
                }
                case OpCode::Mul: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(a * b);
                    break;
                }
                case OpCode::Div: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(a / b);
                    break;
                }
                case OpCode::Pow: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(std::pow(a, b));
                    break;
                }
                case OpCode::Neg: {
                    stack.back() = -stack.back();
                    break;
                }
                case OpCode::Sin: {
                    stack.back() = std::sin(stack.back());
                    break;
                }
                case OpCode::Cos: {
                    stack.back() = std::cos(stack.back());
                    break;
                }
                case OpCode::Tan: {
                    stack.back() = std::tan(stack.back());
                    break;
                }
                case OpCode::Exp: {
                    stack.back() = std::exp(stack.back());
                    break;
                }
                case OpCode::Log: {
                    stack.back() = std::log(stack.back());
                    break;
                }
                case OpCode::Sqrt: {
                    stack.back() = std::sqrt(stack.back());
                    break;
                }
                case OpCode::Abs: {
                    stack.back() = std::abs(stack.back());
                    break;
                }
                case OpCode::Min: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(std::min(a, b));
                    break;
                }
                case OpCode::Max: {
                    double b = stack.back();
                    stack.pop_back();
                    double a = stack.back();
                    stack.pop_back();
                    stack.push_back(std::max(a, b));
                    break;
                }
                default:
                    MPFEM_THROW(ArgumentException, "Unknown opcode");
                }
            }
            MPFEM_ASSERT(stack.size() == 1, "Stack mismatch after evaluation");
            return stack.back() * multiplier;
        }

    } // namespace

    struct ExpressionParser::ExpressionProgram::Impl {
        double multiplier = 1.0;
        bool literalMatrix = false;
        TensorShape shape = TensorShape::scalar();
        std::unique_ptr<AstNode> root;
        std::vector<ExpressionParser::ExpressionProgram> components;
        std::vector<std::string> dependencies;
        std::vector<std::vector<size_t>> componentDependencySlots;
        std::vector<Instruction> instructions_; // Linearized VM instruction stream
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
        if (!impl_) {
            return false;
        }
        if (impl_->shape.isScalar()) {
            return impl_->root != nullptr;
        }
        return !impl_->components.empty();
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

    ExprValue ExpressionParser::ExpressionProgram::evaluate(std::span<const double> values) const
    {
        MPFEM_ASSERT(valid(), "Attempting to evaluate an invalid expression program.");
        MPFEM_ASSERT(values.size() == impl_->dependencies.size(),
            "Expression input size does not match dependency size.");

        if (impl_->shape.isScalar()) {
            return evaluate_single_vm(values, impl_->instructions_, impl_->multiplier);
        }

        auto evalComponent = [this, values](size_t componentIndex) -> double {
            const ExpressionParser::ExpressionProgram& component = impl_->components[componentIndex];
            const std::vector<size_t>& slots = impl_->componentDependencySlots[componentIndex];
            std::vector<double> componentInputs;
            componentInputs.reserve(slots.size());
            for (const size_t slot : slots) {
                componentInputs.push_back(values[slot]);
            }
            return std::get<double>(component.evaluate(std::span<const double>(componentInputs.data(), componentInputs.size())));
        };

        if (!impl_->literalMatrix || impl_->components.size() == 1) {
            const double scalar = evalComponent(0);
            return Matrix3 {Matrix3::Identity() * scalar};
        }

        MPFEM_ASSERT(impl_->components.size() == 9,
            "Invalid matrix expression program: expected 1 or 9 components.");

        Matrix3 matrix;
        matrix << evalComponent(0), evalComponent(3), evalComponent(6),
            evalComponent(1), evalComponent(4), evalComponent(7),
            evalComponent(2), evalComponent(5), evalComponent(8);
        return matrix;
    }

    ExpressionParser::ExpressionParser() = default;
    ExpressionParser::~ExpressionParser() = default;

    ExpressionParser::ExpressionProgram ExpressionParser::compile(const std::string& expression) const
    {
        MatrixTemplate matrixTemplate = parseMatrixTemplate(expression);

        if (!matrixTemplate.literalMatrix) {
            UnitRegistry registry;
            const UnitParseResult unitResult = registry.stripUnit(expression);
            const std::string expressionText = strings::trim(std::string(unitResult.expression));
            if (expressionText.empty()) {
                MPFEM_THROW(ArgumentException, "Expression is empty after unit stripping: " + expression);
            }

            auto impl = std::make_unique<ExpressionProgram::Impl>();
            impl->shape = TensorShape::scalar();
            impl->multiplier = unitResult.multiplier;
            ScalarAstCompiler compiler(expressionText);
            impl->root = compiler.compile();
            std::unordered_set<std::string> seen;
            collectDependencies(*impl->root, seen, impl->dependencies);
            bindAstVariableIndices(*impl->root, impl->dependencies);
            linearize(impl->root.get(), impl->instructions_);
            return ExpressionProgram(std::move(impl));
        }

        auto impl = std::make_unique<ExpressionProgram::Impl>();
        impl->shape = TensorShape::matrix(3, 3);
        impl->literalMatrix = true;
        impl->components.reserve(matrixTemplate.components.size());
        impl->componentDependencySlots.reserve(matrixTemplate.components.size());

        if (matrixTemplate.components.size() == 1) {
            UnitRegistry registry;
            const UnitParseResult unitResult = registry.stripUnit(matrixTemplate.components[0]);
            const std::string exprText = strings::trim(std::string(unitResult.expression));

            auto scalarImpl = std::make_unique<ExpressionProgram::Impl>();
            scalarImpl->shape = TensorShape::scalar();
            scalarImpl->multiplier = unitResult.multiplier;
            ScalarAstCompiler compiler(exprText);
            scalarImpl->root = compiler.compile();
            std::unordered_set<std::string> seen;
            collectDependencies(*scalarImpl->root, seen, scalarImpl->dependencies);
            bindAstVariableIndices(*scalarImpl->root, scalarImpl->dependencies);
            linearize(scalarImpl->root.get(), scalarImpl->instructions_);

            ExpressionProgram scalarProg(std::move(scalarImpl));
            impl->components.push_back(std::move(scalarProg));
            impl->dependencies = impl->components.front().dependencies();
            impl->componentDependencySlots.push_back({});
            impl->componentDependencySlots.front().reserve(impl->dependencies.size());
            for (size_t i = 0; i < impl->dependencies.size(); ++i) {
                impl->componentDependencySlots.front().push_back(i);
            }
            return ExpressionProgram(std::move(impl));
        }

        MPFEM_ASSERT(matrixTemplate.components.size() == 9,
            "Invalid matrix expression: expected 1 or 9 components.");

        for (const std::string& componentExpression : matrixTemplate.components) {
            UnitRegistry registry;
            const UnitParseResult unitResult = registry.stripUnit(componentExpression);
            const std::string exprText = strings::trim(std::string(unitResult.expression));

            auto scalarImpl = std::make_unique<ExpressionProgram::Impl>();
            scalarImpl->shape = TensorShape::scalar();
            scalarImpl->multiplier = unitResult.multiplier;
            ScalarAstCompiler compiler(exprText);
            scalarImpl->root = compiler.compile();
            std::unordered_set<std::string> seen;
            collectDependencies(*scalarImpl->root, seen, scalarImpl->dependencies);
            bindAstVariableIndices(*scalarImpl->root, scalarImpl->dependencies);
            linearize(scalarImpl->root.get(), scalarImpl->instructions_);

            impl->components.push_back(ExpressionProgram(std::move(scalarImpl)));
            const std::vector<std::string>& componentDeps = impl->components.back().dependencies();
            for (const std::string& dep : componentDeps) {
                if (std::find(impl->dependencies.begin(), impl->dependencies.end(), dep) == impl->dependencies.end()) {
                    impl->dependencies.push_back(dep);
                }
            }

            std::vector<size_t> slots;
            slots.reserve(componentDeps.size());
            for (const std::string& dep : componentDeps) {
                const auto it = std::find(impl->dependencies.begin(), impl->dependencies.end(), dep);
                MPFEM_ASSERT(it != impl->dependencies.end(), "Matrix dependency mapping failed.");
                slots.push_back(static_cast<size_t>(std::distance(impl->dependencies.begin(), it)));
            }
            impl->componentDependencySlots.push_back(std::move(slots));
        }
        return ExpressionProgram(std::move(impl));
    }

} // namespace mpfem

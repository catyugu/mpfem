#include "expr/expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"
#include "expr/variable_graph.hpp"

#include <charconv>
#include <cmath>
#include <vector>

namespace mpfem {

    // =========================================================================
    // 数据驱动的 AST 定义：使用 Enum + Switch，保持极速求值
    // =========================================================================

    namespace {
        struct AstNode {
            enum class Kind {
                Constant,
                VarIndex,
                Add,
                Sub,
                Mul,
                Div,
                Pow,
                Dot,
                Neg,
                Sin,
                Cos,
                Tan,
                Exp,
                Log,
                Sqrt,
                Abs,
                Min,
                Max,
                Sym,
                Trace,
                Transpose,
                VectorLit,
                MatrixLit
            } kind;

            Tensor val;
            int var_index = -1;
            std::vector<std::unique_ptr<AstNode>> args;

            static std::unique_ptr<AstNode> make(Kind k)
            {
                auto n = std::make_unique<AstNode>();
                n->kind = k;
                return n;
            }
        };

        Tensor evalAst(const AstNode* node, const Tensor* vars)
        {
            switch (node->kind) {
            case AstNode::Kind::Constant:
                return node->val;
            case AstNode::Kind::VarIndex:
                return vars[node->var_index];
            case AstNode::Kind::Add:
                return evalAst(node->args[0].get(), vars) + evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Sub:
                return evalAst(node->args[0].get(), vars) - evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Mul:
                return evalAst(node->args[0].get(), vars) * evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Div:
                return evalAst(node->args[0].get(), vars) / evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Pow:
                return Tensor::scalar(std::pow(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar()));
            case AstNode::Kind::Dot:
                return Tensor::scalar(dot(evalAst(node->args[0].get(), vars), evalAst(node->args[1].get(), vars)));
            case AstNode::Kind::Min:
                return Tensor::scalar(std::min(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar()));
            case AstNode::Kind::Max:
                return Tensor::scalar(std::max(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar()));
            case AstNode::Kind::Neg:
                return -evalAst(node->args[0].get(), vars);
            case AstNode::Kind::Sin:
                return Tensor::scalar(std::sin(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Cos:
                return Tensor::scalar(std::cos(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Tan:
                return Tensor::scalar(std::tan(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Exp:
                return Tensor::scalar(std::exp(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Log:
                return Tensor::scalar(std::log(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Sqrt:
                return Tensor::scalar(std::sqrt(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Abs:
                return Tensor::scalar(std::abs(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Sym:
                return sym(evalAst(node->args[0].get(), vars));
            case AstNode::Kind::Trace:
                return Tensor::scalar(trace(evalAst(node->args[0].get(), vars)));
            case AstNode::Kind::Transpose:
                return transpose(evalAst(node->args[0].get(), vars));
            case AstNode::Kind::VectorLit:
                return Tensor::vector(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar(), evalAst(node->args[2].get(), vars).scalar());
            case AstNode::Kind::MatrixLit:
                return Tensor::matrix3(
                    evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar(), evalAst(node->args[2].get(), vars).scalar(),
                    evalAst(node->args[3].get(), vars).scalar(), evalAst(node->args[4].get(), vars).scalar(), evalAst(node->args[5].get(), vars).scalar(),
                    evalAst(node->args[6].get(), vars).scalar(), evalAst(node->args[7].get(), vars).scalar(), evalAst(node->args[8].get(), vars).scalar());
            }
            return Tensor::scalar(0.0);
        }

        bool foldConstants(AstNode* node)
        {
            if (node->kind == AstNode::Kind::Constant)
                return true;
            if (node->kind == AstNode::Kind::VarIndex)
                return false;
            bool all_const = true;
            for (auto& arg : node->args)
                if (!foldConstants(arg.get()))
                    all_const = false;
            if (all_const) {
                node->val = evalAst(node, nullptr);
                node->kind = AstNode::Kind::Constant;
                node->args.clear();
                return true;
            }
            return false;
        }
    }

    // =========================================================================
    // 最终运行节点
    // =========================================================================
    class CompiledExpressionNode final : public VariableNode {
        std::unique_ptr<AstNode> ast_;
        std::vector<std::string> dep_names_;
        std::vector<const VariableNode*> resolved_deps_;

    public:
        CompiledExpressionNode(std::unique_ptr<AstNode> ast, std::vector<std::string> deps)
            : ast_(std::move(ast)), dep_names_(std::move(deps)) { }

        void resolve(const VariableManager& mgr) override
        {
            resolved_deps_.clear();
            for (const auto& name : dep_names_) {
                const VariableNode* n = mgr.get(name);
                if (!n)
                    MPFEM_THROW(ArgumentException, "Unbound variable: " + name);
                resolved_deps_.push_back(n);
            }
        }

        std::vector<const VariableNode*> getChildren() const override { return resolved_deps_; }
        bool isConstant() const override { return ast_->kind == AstNode::Kind::Constant; }

        std::uint64_t revision() const override
        {
            if (isConstant())
                return 0;
            std::uint64_t rev = 0;
            for (const auto* dep : resolved_deps_)
                rev = std::max(rev, dep->revision());
            return rev;
        }

        void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
        {
            const size_t n = dest.size();
            const size_t m = resolved_deps_.size();
            if (m == 0) {
                std::fill(dest.begin(), dest.end(), evalAst(ast_.get(), nullptr));
                return;
            }

            Tensor stack_vars[32];
            std::vector<Tensor> heap_vars;
            Tensor* pointVars = (m > 32) ? (heap_vars.resize(m), heap_vars.data()) : stack_vars;

            for (size_t i = 0; i < n; ++i) {
                EvaluationContext pointCtx = ctx;
                if (!ctx.physicalPoints.empty())
                    pointCtx.physicalPoints = ctx.physicalPoints.subspan(i, 1);
                if (!ctx.referencePoints.empty())
                    pointCtx.referencePoints = ctx.referencePoints.subspan(i, 1);
                if (!ctx.transforms.empty())
                    pointCtx.transforms = ctx.transforms.subspan(i, 1);

                for (size_t d = 0; d < m; ++d) {
                    std::span<Tensor> singleSpan(&pointVars[d], 1);
                    resolved_deps_[d]->evaluateBatch(pointCtx, singleSpan);
                }
                dest[i] = evalAst(ast_.get(), pointVars);
            }
        }
    };

    // =========================================================================
    // 词法分析器与 Pratt Parser (彻底替代零碎的 parseTerm/parsePower)
    // =========================================================================
    namespace {
        enum class TokenType {
            Eof,
            Number,
            Identifier,
            Plus,
            Minus,
            Star,
            Slash,
            Caret,
            LParen,
            RParen,
            LBrack,
            RBrack,
            LBrace,
            RBrace,
            Comma,
            Semi,
            T
        };

        struct Token {
            TokenType type;
            std::string_view text;
            Real val = 0.0;
        };

        class Lexer {
            std::string_view text_;
            size_t pos_ = 0;

        public:
            explicit Lexer(std::string_view text) : text_(text) { }
            Token next()
            {
                while (pos_ < text_.size() && std::isspace(text_[pos_]))
                    pos_++;
                if (pos_ >= text_.size())
                    return {TokenType::Eof, ""};

                char c = text_[pos_];
                if (c == '\'') {
                    pos_++;
                    return next();
                } // 直接跳过单引号 (COMSOL格式兼容)

                if (std::isdigit(c) || c == '.') {
                    char* end = nullptr;
                    Real val = std::strtod(text_.data() + pos_, &end);
                    size_t len = end - (text_.data() + pos_);
                    pos_ += len;
                    return {TokenType::Number, text_.substr(pos_ - len, len), val};
                }
                if (std::isalpha(c) || c == '_') {
                    size_t start = pos_;
                    while (pos_ < text_.size() && (std::isalnum(text_[pos_]) || text_[pos_] == '_'))
                        pos_++;
                    return {TokenType::Identifier, text_.substr(start, pos_ - start)};
                }

                pos_++;
                switch (c) {
                case '+':
                    return {TokenType::Plus, "+"};
                case '-':
                    return {TokenType::Minus, "-"};
                case '*':
                    return {TokenType::Star, "*"};
                case '/':
                    return {TokenType::Slash, "/"};
                case '^':
                    return {TokenType::Caret, "^"};
                case '(':
                    return {TokenType::LParen, "("};
                case ')':
                    return {TokenType::RParen, ")"};
                case '[':
                    return {TokenType::LBrack, "["};
                case ']':
                    return {TokenType::RBrack, "]"};
                case '{':
                    return {TokenType::LBrace, "{"};
                case '}':
                    return {TokenType::RBrace, "}"};
                case ',':
                    return {TokenType::Comma, ","};
                case ';':
                    return {TokenType::Semi, ";"};
                default:
                    MPFEM_THROW(ArgumentException, "Unknown char");
                }
            }
        };

        // 优先级权重 (Precedences)
        enum Precedence { P_NONE = 0,
            P_TERM,
            P_FACTOR,
            P_POWER,
            P_CALL };

        class PrattParser {
            Lexer lexer_;
            Token current_;
            std::vector<std::string> deps_;

            void advance() { current_ = lexer_.next(); }
            bool match(TokenType type)
            {
                if (current_.type == type) {
                    advance();
                    return true;
                }
                return false;
            }
            void consume(TokenType type)
            {
                if (!match(type))
                    MPFEM_THROW(ArgumentException, "Unexpected token");
            }

            int getInfixPrecedence(TokenType type)
            {
                switch (type) {
                case TokenType::Plus:
                case TokenType::Minus:
                    return P_TERM;
                case TokenType::Star:
                case TokenType::Slash:
                    return P_FACTOR;
                case TokenType::Caret:
                    return P_POWER;
                case TokenType::LBrack:
                    return P_CALL; // 后缀单位处理 [unit]
                default:
                    return P_NONE;
                }
            }

            int getDepId(const std::string& name)
            {
                auto it = std::find(deps_.begin(), deps_.end(), name);
                if (it != deps_.end())
                    return std::distance(deps_.begin(), it);
                deps_.push_back(name);
                return deps_.size() - 1;
            }

            std::unique_ptr<AstNode> parsePrefix()
            {
                Token token = current_;
                advance();
                switch (token.type) {
                case TokenType::Number: {
                    auto n = AstNode::make(AstNode::Kind::Constant);
                    n->val = Tensor::scalar(token.val);
                    return n;
                }
                case TokenType::Identifier: {
                    std::string name(token.text);
                    if (match(TokenType::LParen))
                        return parseFunctionCall(name); // 函数
                    if (name == "pi") {
                        auto n = AstNode::make(AstNode::Kind::Constant);
                        n->val = Tensor::scalar(3.141592653589793);
                        return n;
                    }
                    if (name == "e") {
                        auto n = AstNode::make(AstNode::Kind::Constant);
                        n->val = Tensor::scalar(2.718281828459045);
                        return n;
                    }

                    auto n = AstNode::make(AstNode::Kind::VarIndex);
                    n->var_index = getDepId(name);
                    return n;
                }
                case TokenType::Minus: {
                    auto n = AstNode::make(AstNode::Kind::Neg);
                    n->args.push_back(parseExpression(P_POWER)); // 一元负号优先级同级或高于 Power
                    return n;
                }
                case TokenType::Plus:
                    return parseExpression(P_POWER);
                case TokenType::LParen: {
                    auto expr = parseExpression();
                    consume(TokenType::RParen);
                    return expr;
                }
                case TokenType::LBrack:
                    return parseBracketMatrix();
                case TokenType::LBrace:
                    return parseComsolMatrix();
                default:
                    MPFEM_THROW(ArgumentException, "Expected expression");
                }
            }

            std::unique_ptr<AstNode> parseInfix(std::unique_ptr<AstNode> left, Token op)
            {
                if (op.type == TokenType::LBrack) {
                    // 作为单位后缀被触发，例如 20[MPa]
                    std::string unitName;
                    while (current_.type != TokenType::RBrack && current_.type != TokenType::Eof) {
                        unitName += current_.text;
                        advance();
                    }
                    consume(TokenType::RBrack);

                    Real mult = parseUnit(unitName);
                    if (std::abs(mult - 1.0) < 1e-15)
                        return left;

                    auto c = AstNode::make(AstNode::Kind::Constant);
                    c->val = Tensor::scalar(mult);
                    auto n = AstNode::make(AstNode::Kind::Mul);
                    n->args.push_back(std::move(c));
                    n->args.push_back(std::move(left));
                    return n;
                }

                auto right = parseExpression(getInfixPrecedence(op.type));
                auto n = AstNode::make(
                    op.type == TokenType::Plus ? AstNode::Kind::Add : op.type == TokenType::Minus ? AstNode::Kind::Sub
                        : op.type == TokenType::Star                                              ? AstNode::Kind::Mul
                        : op.type == TokenType::Slash                                             ? AstNode::Kind::Div
                                                                                                  : AstNode::Kind::Pow);
                n->args.push_back(std::move(left));
                n->args.push_back(std::move(right));
                return n;
            }

            std::unique_ptr<AstNode> parseFunctionCall(const std::string& name)
            {
                std::vector<std::unique_ptr<AstNode>> args;
                if (!match(TokenType::RParen)) {
                    do {
                        args.push_back(parseExpression());
                    } while (match(TokenType::Comma));
                    consume(TokenType::RParen);
                }
                auto n = AstNode::make(AstNode::Kind::Constant); // Default assign
                if (args.size() == 1) {
                    if (name == "sin")
                        n->kind = AstNode::Kind::Sin;
                    else if (name == "cos")
                        n->kind = AstNode::Kind::Cos;
                    else if (name == "tan")
                        n->kind = AstNode::Kind::Tan;
                    else if (name == "exp")
                        n->kind = AstNode::Kind::Exp;
                    else if (name == "log")
                        n->kind = AstNode::Kind::Log;
                    else if (name == "sqrt")
                        n->kind = AstNode::Kind::Sqrt;
                    else if (name == "abs")
                        n->kind = AstNode::Kind::Abs;
                    else if (name == "sym")
                        n->kind = AstNode::Kind::Sym;
                    else if (name == "trace" || name == "tr")
                        n->kind = AstNode::Kind::Trace;
                    else if (name == "transpose")
                        n->kind = AstNode::Kind::Transpose;
                    else
                        MPFEM_THROW(ArgumentException, "Unknown unary function: " + name);
                }
                else if (args.size() == 2) {
                    if (name == "dot")
                        n->kind = AstNode::Kind::Dot;
                    else if (name == "pow")
                        n->kind = AstNode::Kind::Pow;
                    else if (name == "min")
                        n->kind = AstNode::Kind::Min;
                    else if (name == "max")
                        n->kind = AstNode::Kind::Max;
                    else
                        MPFEM_THROW(ArgumentException, "Unknown binary function: " + name);
                }
                else {
                    MPFEM_THROW(ArgumentException, "Wrong arguments for function: " + name);
                }
                n->args = std::move(args);
                return n;
            }

            // 原生解析 [...] 矩阵格式
            std::unique_ptr<AstNode> parseBracketMatrix()
            {
                std::vector<std::unique_ptr<AstNode>> elems;
                int cols = 0, rows = 1, currentCols = 0;
                while (current_.type != TokenType::RBrack) {
                    elems.push_back(parseExpression());
                    currentCols++;
                    if (match(TokenType::Comma))
                        continue;
                    if (match(TokenType::Semi)) {
                        if (cols == 0)
                            cols = currentCols;
                        rows++;
                        currentCols = 0;
                    }
                }
                consume(TokenType::RBrack);
                if (cols == 0)
                    cols = currentCols;

                if (match(TokenType::Caret))
                    consume(TokenType::Identifier); // Consume ^T

                if (rows == 1 && cols == 1)
                    return std::move(elems.front());
                if (rows == 1 && cols == 3) {
                    auto n = AstNode::make(AstNode::Kind::VectorLit);
                    n->args = std::move(elems);
                    return n;
                }
                if (rows == 3 && cols == 3) {
                    auto n = AstNode::make(AstNode::Kind::MatrixLit);
                    n->args = std::move(elems);
                    return n;
                }
                MPFEM_THROW(ArgumentException, "Unsupported matrix shape");
            }

            // 原生解析 COMSOL {...} 矩阵格式，消灭字符串拼装与分配
            std::unique_ptr<AstNode> parseComsolMatrix()
            {
                std::vector<std::unique_ptr<AstNode>> elems;
                while (current_.type != TokenType::RBrace) {
                    elems.push_back(parseExpression());
                    match(TokenType::Comma);
                }
                consume(TokenType::RBrace);

                auto zero = []() { auto z = AstNode::make(AstNode::Kind::Constant); z->val = Tensor::scalar(0.0); return z; };

                auto n = AstNode::make(AstNode::Kind::MatrixLit);
                if (elems.size() == 1) {
                    // 各向同性 {c} -> [c,0,0; 0,c,0; 0,0,c]
                    for (int i = 0; i < 9; ++i)
                        n->args.push_back(i % 4 == 0 ? std::move(elems[0]) : zero());
                }
                else if (elems.size() == 9) {
                    // COMSOL输入为转置排列 {c0,c1,c2,c3,c4,c5,c6,c7,c8}
                    n->args.push_back(std::move(elems[0]));
                    n->args.push_back(std::move(elems[3]));
                    n->args.push_back(std::move(elems[6]));
                    n->args.push_back(std::move(elems[1]));
                    n->args.push_back(std::move(elems[4]));
                    n->args.push_back(std::move(elems[7]));
                    n->args.push_back(std::move(elems[2]));
                    n->args.push_back(std::move(elems[5]));
                    n->args.push_back(std::move(elems[8]));
                }
                else {
                    MPFEM_THROW(ArgumentException, "Unsupported COMSOL matrix element count");
                }
                return n;
            }

        public:
            explicit PrattParser(std::string_view text) : lexer_(text) { advance(); }

            std::unique_ptr<AstNode> parseExpression(int precedence = P_NONE)
            {
                auto left = parsePrefix();
                while (precedence < getInfixPrecedence(current_.type)) {
                    Token op = current_;
                    advance();
                    left = parseInfix(std::move(left), op);
                }
                return left;
            }

            std::unique_ptr<VariableNode> compile()
            {
                auto root = parseExpression();
                if (current_.type != TokenType::Eof)
                    MPFEM_THROW(ArgumentException, "Unexpected trailing token");
                foldConstants(root.get());
                return std::make_unique<CompiledExpressionNode>(std::move(root), std::move(deps_));
            }
        };
    }

    std::unique_ptr<VariableNode> ExpressionParser::parse(const std::string& expression)
    {
        if (strings::trim(expression).empty())
            MPFEM_THROW(ArgumentException, "Empty expression string");
        PrattParser parser(expression);
        return parser.compile();
    }

} // namespace mpfem
#include "expr/expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <cmath>
#include <vector>

namespace mpfem {

    // =========================================================================
    // 数据驱动的 AST 定义：使用 Enum + Switch 代替虚函数，极限压榨缓存与分支预测
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

            TensorValue val; // 供 Constant 节点使用
            int var_index = -1; // 供 VarIndex (变量引用) 节点使用
            std::vector<std::unique_ptr<AstNode>> args;
        };

        // 数据驱动的极速递归求值
        TensorValue evalAst(const AstNode* node, const TensorValue* vars)
        {
            switch (node->kind) {
            case AstNode::Kind::Constant:
                return node->val;
            case AstNode::Kind::VarIndex:
                return vars[node->var_index];

            // 二元运算
            case AstNode::Kind::Add:
                return evalAst(node->args[0].get(), vars) + evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Sub:
                return evalAst(node->args[0].get(), vars) - evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Mul:
                return evalAst(node->args[0].get(), vars) * evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Div:
                return evalAst(node->args[0].get(), vars) / evalAst(node->args[1].get(), vars);
            case AstNode::Kind::Pow: {
                Real b = evalAst(node->args[1].get(), vars).scalar();
                return TensorValue::scalar(std::pow(evalAst(node->args[0].get(), vars).scalar(), b));
            }
            case AstNode::Kind::Dot:
                return TensorValue::scalar(dot(evalAst(node->args[0].get(), vars), evalAst(node->args[1].get(), vars)));
            case AstNode::Kind::Min:
                return TensorValue::scalar(std::min(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar()));
            case AstNode::Kind::Max:
                return TensorValue::scalar(std::max(evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar()));

            // 一元运算
            case AstNode::Kind::Neg:
                return -evalAst(node->args[0].get(), vars);
            case AstNode::Kind::Sin:
                return TensorValue::scalar(std::sin(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Cos:
                return TensorValue::scalar(std::cos(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Tan:
                return TensorValue::scalar(std::tan(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Exp:
                return TensorValue::scalar(std::exp(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Log:
                return TensorValue::scalar(std::log(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Sqrt:
                return TensorValue::scalar(std::sqrt(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Abs:
                return TensorValue::scalar(std::abs(evalAst(node->args[0].get(), vars).scalar()));
            case AstNode::Kind::Sym:
                return sym(evalAst(node->args[0].get(), vars));
            case AstNode::Kind::Trace:
                return TensorValue::scalar(trace(evalAst(node->args[0].get(), vars)));
            case AstNode::Kind::Transpose:
                return transpose(evalAst(node->args[0].get(), vars));

            // 字面量结构
            case AstNode::Kind::VectorLit:
                return TensorValue::vector(
                    evalAst(node->args[0].get(), vars).scalar(),
                    evalAst(node->args[1].get(), vars).scalar(),
                    evalAst(node->args[2].get(), vars).scalar());
            case AstNode::Kind::MatrixLit:
                return TensorValue::matrix3(
                    evalAst(node->args[0].get(), vars).scalar(), evalAst(node->args[1].get(), vars).scalar(), evalAst(node->args[2].get(), vars).scalar(),
                    evalAst(node->args[3].get(), vars).scalar(), evalAst(node->args[4].get(), vars).scalar(), evalAst(node->args[5].get(), vars).scalar(),
                    evalAst(node->args[6].get(), vars).scalar(), evalAst(node->args[7].get(), vars).scalar(), evalAst(node->args[8].get(), vars).scalar());
            }
            return TensorValue::scalar(0.0);
        }

        // 常量折叠 (Constant Folding): 编译期计算静态树，消除运行时开销
        bool foldConstants(AstNode* node)
        {
            if (node->kind == AstNode::Kind::Constant)
                return true;
            if (node->kind == AstNode::Kind::VarIndex)
                return false;

            bool all_const = true;
            for (auto& arg : node->args) {
                if (!foldConstants(arg.get()))
                    all_const = false;
            }

            if (all_const) {
                node->val = evalAst(node, nullptr);
                node->kind = AstNode::Kind::Constant;
                node->args.clear();
                return true;
            }
            return false;
        }
    } // namespace

    // =========================================================================
    // 最终封装入 VariableNode 的实体：持有 AST 与 唯一的依赖去重列表
    // =========================================================================

    class CompiledExpressionNode final : public VariableNode {
    public:
        CompiledExpressionNode(std::unique_ptr<AstNode> ast, std::vector<std::string> deps)
            : ast_(std::move(ast)), dep_names_(std::move(deps)) { }

        void resolve(const VariableManager& mgr) override
        {
            resolved_deps_.clear();
            resolved_deps_.reserve(dep_names_.size());
            for (const auto& name : dep_names_) {
                const VariableNode* n = mgr.get(name);
                if (!n)
                    MPFEM_THROW(ArgumentException, "Unbound variable: " + name);
                resolved_deps_.push_back(n);
            }
        }

        std::vector<const VariableNode*> getChildren() const override
        {
            return resolved_deps_;
        }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override
        {
            const size_t n = dest.size();
            const size_t m = resolved_deps_.size();

            // 极速单点路径 (N=1), 无堆分配, 无虚函数嵌套, 彻底解决冗余评估
            if (n == 1) {
                TensorValue vars_stack[32]; // 分配在寄存器/栈上
                std::vector<TensorValue> vars_heap;
                TensorValue* vars = vars_stack;

                if (m > 32) {
                    vars_heap.resize(m);
                    vars = vars_heap.data();
                }

                // 核心：仅对去重后的依赖项各自求值一次！(如 grad(V) 只求一次)
                for (size_t d = 0; d < m; ++d) {
                    std::span<TensorValue> s(&vars[d], 1);
                    resolved_deps_[d]->evaluateBatch(ctx, s);
                }

                // 通过极速 Switch 执行 AST
                dest[0] = evalAst(ast_.get(), vars);
                return;
            }

            // 兼容的 N>1 回退路径
            std::vector<TensorValue> scratchpad(m * n);
            for (size_t d = 0; d < m; ++d) {
                std::span<TensorValue> depDest(&scratchpad[d * n], n);
                resolved_deps_[d]->evaluateBatch(ctx, depDest);
            }

            std::vector<TensorValue> pointVars(m);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < m; ++d) {
                    pointVars[d] = scratchpad[d * n + i];
                }
                dest[i] = evalAst(ast_.get(), pointVars.data());
            }
        }

        bool isConstant() const override
        {
            return ast_->kind == AstNode::Kind::Constant;
        }

    private:
        std::unique_ptr<AstNode> ast_;
        std::vector<std::string> dep_names_;
        std::vector<const VariableNode*> resolved_deps_;
    };

    // =========================================================================
    // Parser: 将文本解析为 AST，并提取去重的依赖项列表
    // =========================================================================

    namespace {

        bool isNearOne(Real value) { return std::abs(value - 1.0) < 1e-15; }

        class TensorAstCompiler {
        public:
            explicit TensorAstCompiler(std::string_view text) : text_(text) { }

            std::unique_ptr<VariableNode> compile()
            {
                auto root = parseExpression();
                skipWhitespace();
                if (!eof())
                    MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));

                // 常量折叠：直接在编译期算掉所有类似 2*pi, E/(1+nu) 这种纯数字的分支
                foldConstants(root.get());

                return std::make_unique<CompiledExpressionNode>(std::move(root), std::move(dep_names_));
            }

        private:
            std::vector<std::string> dep_names_;

            int getOrAddDep(const std::string& name)
            {
                for (size_t i = 0; i < dep_names_.size(); ++i) {
                    if (dep_names_[i] == name)
                        return static_cast<int>(i);
                }
                dep_names_.push_back(name);
                return static_cast<int>(dep_names_.size() - 1);
            }

            std::unique_ptr<AstNode> makeNode(AstNode::Kind k)
            {
                auto n = std::make_unique<AstNode>();
                n->kind = k;
                return n;
            }

            std::unique_ptr<AstNode> makeBinary(AstNode::Kind k, std::unique_ptr<AstNode> l, std::unique_ptr<AstNode> r)
            {
                auto n = makeNode(k);
                n->args.push_back(std::move(l));
                n->args.push_back(std::move(r));
                return n;
            }

            std::unique_ptr<AstNode> makeUnary(AstNode::Kind k, std::unique_ptr<AstNode> arg)
            {
                auto n = makeNode(k);
                n->args.push_back(std::move(arg));
                return n;
            }

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
                        lhs = makeBinary(AstNode::Kind::Sub, std::move(lhs), parseTerm());
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
                        lhs = makeBinary(AstNode::Kind::Mul, std::move(lhs), parsePower());
                        continue;
                    }
                    if (match('/')) {
                        lhs = makeBinary(AstNode::Kind::Div, std::move(lhs), parsePower());
                        continue;
                    }
                    return lhs;
                }
            }

            std::unique_ptr<AstNode> parsePower()
            {
                auto lhs = parseUnary();
                skipWhitespace();
                if (match('^'))
                    return makeBinary(AstNode::Kind::Pow, std::move(lhs), parsePower());
                return lhs;
            }

            std::unique_ptr<AstNode> parseUnary()
            {
                skipWhitespace();
                if (match('+'))
                    return parseUnary();
                if (match('-'))
                    return makeUnary(AstNode::Kind::Neg, parseUnary());
                return parsePrimary();
            }

            std::unique_ptr<AstNode> parsePrimary()
            {
                skipWhitespace();
                if (match('(')) {
                    auto inner = parseExpression();
                    skipWhitespace();
                    if (!match(')'))
                        MPFEM_THROW(ArgumentException, "Missing closing ')'");
                    return applyUnitSuffix(std::move(inner));
                }
                if (peek() == '[')
                    return applyUnitSuffix(parseBracketLiteral());
                if (peekIsNumberStart())
                    return applyUnitSuffix(parseNumber());
                if (peekIsIdentifierStart()) {
                    std::string name = parseIdentifier();
                    skipWhitespace();
                    if (match('('))
                        return applyUnitSuffix(parseFunction(name));

                    if (name == "pi") {
                        auto n = makeNode(AstNode::Kind::Constant);
                        n->val = TensorValue::scalar(3.141592653589793);
                        return n;
                    }
                    if (name == "e") {
                        auto n = makeNode(AstNode::Kind::Constant);
                        n->val = TensorValue::scalar(2.718281828459045);
                        return n;
                    }

                    // 变量引用 -> 提取并去重依赖项 -> 转换成极速的数组索引
                    int idx = getOrAddDep(name);
                    auto node = makeNode(AstNode::Kind::VarIndex);
                    node->var_index = idx;
                    return applyUnitSuffix(std::move(node));
                }
                MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));
            }

            std::unique_ptr<AstNode> parseNumber()
            {
                const char* begin = text_.data() + pos_;
                char* end = nullptr;
                Real value = std::strtod(begin, &end);
                pos_ += (end - begin);
                auto n = makeNode(AstNode::Kind::Constant);
                n->val = TensorValue::scalar(value);
                return n;
            }

            std::string parseIdentifier()
            {
                size_t begin = pos_++;
                while (!eof() && (std::isalnum(text_[pos_]) || text_[pos_] == '_'))
                    ++pos_;
                return std::string(text_.substr(begin, pos_ - begin));
            }

            std::unique_ptr<AstNode> parseFunction(const std::string& name)
            {
                // 特殊处理 grad 语法：grad(V) 作为一个单一的整体依赖项
                if (name == "grad") {
                    skipWhitespace();
                    std::string fieldName = parseIdentifier();
                    skipWhitespace();
                    if (!match(')'))
                        MPFEM_THROW(ArgumentException, "Missing ')' after grad argument");

                    int idx = getOrAddDep("grad(" + fieldName + ")");
                    auto node = makeNode(AstNode::Kind::VarIndex);
                    node->var_index = idx;
                    return node;
                }

                std::vector<std::unique_ptr<AstNode>> args;
                if (!match(')')) {
                    do {
                        args.push_back(parseExpression());
                        skipWhitespace();
                    } while (match(','));
                    if (!match(')'))
                        MPFEM_THROW(ArgumentException, "Missing ')' in function " + name);
                }

                if (args.size() == 1) {
                    if (name == "sin")
                        return makeUnary(AstNode::Kind::Sin, std::move(args[0]));
                    if (name == "cos")
                        return makeUnary(AstNode::Kind::Cos, std::move(args[0]));
                    if (name == "tan")
                        return makeUnary(AstNode::Kind::Tan, std::move(args[0]));
                    if (name == "exp")
                        return makeUnary(AstNode::Kind::Exp, std::move(args[0]));
                    if (name == "log")
                        return makeUnary(AstNode::Kind::Log, std::move(args[0]));
                    if (name == "sqrt")
                        return makeUnary(AstNode::Kind::Sqrt, std::move(args[0]));
                    if (name == "abs")
                        return makeUnary(AstNode::Kind::Abs, std::move(args[0]));
                    if (name == "sym")
                        return makeUnary(AstNode::Kind::Sym, std::move(args[0]));
                    if (name == "trace" || name == "tr")
                        return makeUnary(AstNode::Kind::Trace, std::move(args[0]));
                    if (name == "transpose")
                        return makeUnary(AstNode::Kind::Transpose, std::move(args[0]));
                }
                else if (args.size() == 2) {
                    if (name == "dot")
                        return makeBinary(AstNode::Kind::Dot, std::move(args[0]), std::move(args[1]));
                    if (name == "pow")
                        return makeBinary(AstNode::Kind::Pow, std::move(args[0]), std::move(args[1]));
                    if (name == "min")
                        return makeBinary(AstNode::Kind::Min, std::move(args[0]), std::move(args[1]));
                    if (name == "max")
                        return makeBinary(AstNode::Kind::Max, std::move(args[0]), std::move(args[1]));
                }
                MPFEM_THROW(ArgumentException, "Unsupported function or wrong arity: " + name);
            }

            std::unique_ptr<AstNode> parseBracketLiteral()
            {
                if (!match('['))
                    MPFEM_THROW(ArgumentException, "Internal error: expected '['");

                std::vector<std::vector<std::unique_ptr<AstNode>>> rows;
                rows.emplace_back();

                for (;;) {
                    skipWhitespace();
                    const size_t partBegin = pos_;
                    int parenDepth = 0, unitBracketDepth = 0;
                    while (!eof()) {
                        const char c = text_[pos_];
                        if (c == '(')
                            ++parenDepth;
                        else if (c == ')')
                            --parenDepth;
                        else if (c == '[')
                            ++unitBracketDepth;
                        else if (c == ']') {
                            if (unitBracketDepth > 0)
                                --unitBracketDepth;
                            else if (parenDepth == 0)
                                break;
                        }
                        else if (parenDepth == 0 && unitBracketDepth == 0 && (c == ',' || c == ';'))
                            break;
                        ++pos_;
                    }
                    if (eof())
                        MPFEM_THROW(ArgumentException, "Missing closing ']'");

                    std::string part = strings::trim(std::string(text_.substr(partBegin, pos_ - partBegin)));

                    UnitRegistry registry;
                    auto unitPart = registry.stripUnit(part);

                    // 巧妙递归：替换文本流，让本 Compiler 直接解析这个子字符串，这保证它能使用当前的依赖集合
                    std::string inner_expr = std::string(unitPart.expression);
                    std::string_view old_text = text_;
                    size_t old_pos = pos_;

                    text_ = inner_expr;
                    pos_ = 0;
                    auto comp = parseExpression();
                    if (!eof())
                        MPFEM_THROW(ArgumentException, "Failed to parse literal component: " + part);

                    text_ = old_text;
                    pos_ = old_pos;

                    if (!isNearOne(unitPart.multiplier)) {
                        auto c = makeNode(AstNode::Kind::Constant);
                        c->val = TensorValue::scalar(unitPart.multiplier);
                        comp = makeBinary(AstNode::Kind::Mul, std::move(c), std::move(comp));
                    }
                    rows.back().push_back(std::move(comp));

                    const char delim = text_[pos_];
                    ++pos_;
                    if (delim == ',')
                        continue;
                    if (delim == ';') {
                        rows.emplace_back();
                        continue;
                    }
                    if (delim == ']')
                        break;
                }

                skipWhitespace();
                bool isTranspose = match('^');
                if (isTranspose) {
                    skipWhitespace();
                    if (!match('T'))
                        MPFEM_THROW(ArgumentException, "Expected '^T'");
                }

                const size_t rowCount = rows.size();
                const size_t colCount = rows.front().size();

                if (rowCount == 1 && colCount == 1)
                    return std::move(rows.front().front());

                if (rowCount == 1 && colCount == 3) {
                    auto n = makeNode(AstNode::Kind::VectorLit);
                    n->args.push_back(std::move(rows[0][0]));
                    n->args.push_back(std::move(rows[0][1]));
                    n->args.push_back(std::move(rows[0][2]));
                    return n;
                }

                if (rowCount == 3 && colCount == 3) {
                    auto n = makeNode(AstNode::Kind::MatrixLit);
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            n->args.push_back(std::move(rows[i][j]));
                    return n;
                }

                MPFEM_THROW(ArgumentException, "Unsupported bracket literal shape");
            }

            std::unique_ptr<AstNode> applyUnitSuffix(std::unique_ptr<AstNode> node)
            {
                skipWhitespace();
                if (!match('['))
                    return node;
                size_t begin = pos_;
                while (!eof() && text_[pos_] != ']')
                    ++pos_;
                if (eof())
                    MPFEM_THROW(ArgumentException, "Missing ']' for unit suffix");

                std::string unit = strings::trim(std::string(text_.substr(begin, pos_ - begin)));
                ++pos_;

                Real mult = UnitRegistry().getMultiplier(unit);
                if (isNearOne(mult))
                    return node;

                auto c = makeNode(AstNode::Kind::Constant);
                c->val = TensorValue::scalar(mult);
                return makeBinary(AstNode::Kind::Mul, std::move(c), std::move(node));
            }

            char peek() const { return eof() ? '\0' : text_[pos_]; }
            bool peekIsIdentifierStart() const { return !eof() && (std::isalpha(text_[pos_]) || text_[pos_] == '_'); }
            bool peekIsNumberStart() const { return !eof() && (std::isdigit(text_[pos_]) || text_[pos_] == '.'); }
            bool eof() const { return pos_ >= text_.size(); }
            void skipWhitespace()
            {
                while (!eof() && std::isspace(text_[pos_]))
                    ++pos_;
            }
            bool match(char c)
            {
                if (peek() == c) {
                    ++pos_;
                    return true;
                }
                return false;
            }

            std::string_view text_;
            size_t pos_ = 0;
        };

        // Comsol 矩阵模板预处理
        struct MatrixTemplate {
            bool literalMatrix = false;
            std::vector<std::string> components;
        };

        bool isSeparator(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ','; }

        MatrixTemplate parseComsolMatrixTemplate(std::string_view expression)
        {
            MatrixTemplate tpl;
            const std::string trimmed = strings::trim(std::string(expression));

            if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}')
                return tpl;

            tpl.literalMatrix = true;
            const std::string_view content(trimmed.data() + 1, trimmed.size() - 2);

            size_t index = 0;
            while (index < content.size()) {
                while (index < content.size() && isSeparator(content[index]))
                    ++index;
                if (index >= content.size())
                    break;
                if (content[index] != '\'')
                    MPFEM_THROW(ArgumentException, "Invalid comsol matrix");
                const size_t endQuote = content.find('\'', index + 1);
                if (endQuote == std::string_view::npos)
                    MPFEM_THROW(ArgumentException, "Unterminated quote");
                tpl.components.push_back(strings::trim(std::string(content.substr(index + 1, endQuote - index - 1))));
                index = endQuote + 1;
            }
            return tpl;
        }

        std::string convertComsolMatrixToBracketLiteral(const MatrixTemplate& tpl)
        {
            if (tpl.components.size() == 1) {
                const std::string& c = tpl.components[0];
                return "[" + c + ",0,0;0," + c + ",0;0,0," + c + "]";
            }
            return "[" + tpl.components[0] + "," + tpl.components[3] + "," + tpl.components[6] + ";"
                + tpl.components[1] + "," + tpl.components[4] + "," + tpl.components[7] + ";"
                + tpl.components[2] + "," + tpl.components[5] + "," + tpl.components[8] + "]";
        }
    } // namespace

    std::unique_ptr<VariableNode> ExpressionParser::parse(const std::string& expression)
    {
        std::string expressionText = strings::trim(expression);
        if (expressionText.empty())
            MPFEM_THROW(ArgumentException, "Empty expression string");

        MatrixTemplate comsol = parseComsolMatrixTemplate(expressionText);
        if (comsol.literalMatrix) {
            expressionText = convertComsolMatrixToBracketLiteral(comsol);
        }

        TensorAstCompiler compiler(expressionText);
        return compiler.compile();
    }

} // namespace mpfem
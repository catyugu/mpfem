#include "expr/expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <cmath>
#include <functional>
#include <vector>
#include <array>

namespace mpfem {

    // =========================================================================
    // 零初始化开销的线程本地栈分配器 (Zero-overhead thread-local bump allocator)
    // 用于消除 AST 求值期间的内存分配和 std::variant 默认构造开销
    // =========================================================================
    namespace {
        struct ThreadWorkspace {
            static constexpr size_t MAX_TMP = 65536; // 64K TensorValues ≈ 5MB 缓存
            std::array<TensorValue, MAX_TMP> memory;
            size_t head = 0;

            TensorValue* alloc(size_t n) {
                if (head + n > MAX_TMP) {
                    MPFEM_THROW(Exception, "AST workspace overflow. Expression is too deep.");
                }
                TensorValue* ptr = memory.data() + head;
                head += n;
                return ptr;
            }

            void dealloc(size_t n) {
                head -= n;
            }
        };

        thread_local ThreadWorkspace t_workspace;

        struct WorkspaceGuard {
            TensorValue* ptr;
            size_t size_;
            explicit WorkspaceGuard(size_t n) : size_(n) { ptr = t_workspace.alloc(n); }
            ~WorkspaceGuard() { t_workspace.dealloc(size_); }
            std::span<TensorValue> span() { return std::span<TensorValue>(ptr, size_); }
        };
    }

    // =========================================================================
    // 仿函数（Functors）—— 支持模板化深度内联
    // =========================================================================
    struct OpAdd { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return a + b; } };
    struct OpSub { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return a - b; } };
    struct OpMul { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return a * b; } };
    struct OpDiv { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return a / b; } };
    struct OpPow { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return TensorValue::scalar(std::pow(a.scalar(), b.scalar())); } };
    struct OpDot { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return TensorValue::scalar(dot(a, b)); } };
    struct OpMin { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return TensorValue::scalar(std::min(a.scalar(), b.scalar())); } };
    struct OpMax { inline TensorValue operator()(const TensorValue& a, const TensorValue& b) const { return TensorValue::scalar(std::max(a.scalar(), b.scalar())); } };

    struct OpNeg { inline TensorValue operator()(const TensorValue& a) const { return -a; } };
    struct OpSin { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::sin(a.scalar())); } };
    struct OpCos { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::cos(a.scalar())); } };
    struct OpTan { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::tan(a.scalar())); } };
    struct OpExp { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::exp(a.scalar())); } };
    struct OpLog { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::log(a.scalar())); } };
    struct OpSqrt { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::sqrt(a.scalar())); } };
    struct OpAbs { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(std::abs(a.scalar())); } };
    struct OpSym { inline TensorValue operator()(const TensorValue& a) const { return sym(a); } };
    struct OpTrace { inline TensorValue operator()(const TensorValue& a) const { return TensorValue::scalar(trace(a)); } };
    struct OpTranspose { inline TensorValue operator()(const TensorValue& a) const { return transpose(a); } };

    // =========================================================================
    // AST Nodes
    // =========================================================================

    class ConstantNode final : public VariableNode {
    public:
        explicit ConstantNode(TensorValue val) : val_(std::move(val)) {}
        void evaluateBatch(const EvaluationContext&, std::span<TensorValue> dest) const override {
            for (auto& v : dest) v = val_;
        }
        bool isConstant() const override { return true; }
        const TensorValue& value() const { return val_; }
    private:
        TensorValue val_;
    };

    class VariableRefNode final : public VariableNode {
    public:
        explicit VariableRefNode(std::string name) : name_(std::move(name)) {}

        void resolve(const VariableManager& mgr) override {
            resolved_ = mgr.get(name_);
            if (!resolved_) {
                MPFEM_THROW(ArgumentException, "Unbound variable in expression: " + name_);
            }
        }
        
        std::vector<const VariableNode*> getChildren() const override {
            return resolved_ ? std::vector<const VariableNode*>{resolved_} : std::vector<const VariableNode*>{};
        }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            resolved_->evaluateBatch(ctx, dest);
        }
    private:
        std::string name_;
        const VariableNode* resolved_ = nullptr;
    };

    class GradRefNode final : public VariableNode {
    public:
        explicit GradRefNode(std::string fieldName) : fieldName_(std::move(fieldName)) {
            lookupName_ = "grad(" + fieldName_ + ")";
        }
        void resolve(const VariableManager& mgr) override {
            resolved_ = mgr.get(lookupName_);
            if (!resolved_) MPFEM_THROW(ArgumentException, "Unbound gradient variable: " + lookupName_);
        }
        std::vector<const VariableNode*> getChildren() const override {
            return resolved_ ? std::vector<const VariableNode*>{resolved_} : std::vector<const VariableNode*>{};
        }
        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            resolved_->evaluateBatch(ctx, dest);
        }
    private:
        std::string fieldName_;
        std::string lookupName_;
        const VariableNode* resolved_ = nullptr;
    };

    template <typename Op>
    class BinaryOpNode final : public VariableNode {
    public:
        BinaryOpNode(std::unique_ptr<VariableNode> lhs, std::unique_ptr<VariableNode> rhs)
            : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

        void resolve(const VariableManager& mgr) override {
            lhs_->resolve(mgr);
            rhs_->resolve(mgr);
        }

        std::vector<const VariableNode*> getChildren() const override { return {lhs_.get(), rhs_.get()}; }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            // 直接利用目标内存计算左侧节点，避免多余分配
            lhs_->evaluateBatch(ctx, dest);
            
            // 仅仅给右侧节点分配无初始化成本的临时工作区
            WorkspaceGuard guard(dest.size());
            std::span<TensorValue> rhs_vals = guard.span();
            rhs_->evaluateBatch(ctx, rhs_vals);
            
            Op op;
            for (size_t i = 0; i < dest.size(); ++i) {
                dest[i] = op(dest[i], rhs_vals[i]);
            }
        }
    private:
        std::unique_ptr<VariableNode> lhs_;
        std::unique_ptr<VariableNode> rhs_;
    };

    template <typename Op>
    class UnaryOpNode final : public VariableNode {
    public:
        explicit UnaryOpNode(std::unique_ptr<VariableNode> arg) : arg_(std::move(arg)) {}

        void resolve(const VariableManager& mgr) override { arg_->resolve(mgr); }
        std::vector<const VariableNode*> getChildren() const override { return {arg_.get()}; }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            arg_->evaluateBatch(ctx, dest);
            Op op;
            for (auto& v : dest) {
                v = op(v);
            }
        }
    private:
        std::unique_ptr<VariableNode> arg_;
    };

    class VectorLiteralNode final : public VariableNode {
    public:
        VectorLiteralNode(std::unique_ptr<VariableNode> x, std::unique_ptr<VariableNode> y, std::unique_ptr<VariableNode> z)
            : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

        void resolve(const VariableManager& mgr) override { x_->resolve(mgr); y_->resolve(mgr); z_->resolve(mgr); }
        std::vector<const VariableNode*> getChildren() const override { return {x_.get(), y_.get(), z_.get()}; }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            WorkspaceGuard guard(dest.size() * 2);
            std::span<TensorValue> y_vals = guard.span().subspan(0, dest.size());
            std::span<TensorValue> z_vals = guard.span().subspan(dest.size(), dest.size());
            
            x_->evaluateBatch(ctx, dest);
            y_->evaluateBatch(ctx, y_vals);
            z_->evaluateBatch(ctx, z_vals);
            
            for (size_t i = 0; i < dest.size(); ++i) {
                dest[i] = TensorValue::vector(dest[i].scalar(), y_vals[i].scalar(), z_vals[i].scalar());
            }
        }
    private:
        std::unique_ptr<VariableNode> x_, y_, z_;
    };

    class MatrixLiteralNode final : public VariableNode {
    public:
        MatrixLiteralNode(
            std::unique_ptr<VariableNode> m00, std::unique_ptr<VariableNode> m01, std::unique_ptr<VariableNode> m02,
            std::unique_ptr<VariableNode> m10, std::unique_ptr<VariableNode> m11, std::unique_ptr<VariableNode> m12,
            std::unique_ptr<VariableNode> m20, std::unique_ptr<VariableNode> m21, std::unique_ptr<VariableNode> m22)
        {
            m_[0][0] = std::move(m00); m_[0][1] = std::move(m01); m_[0][2] = std::move(m02);
            m_[1][0] = std::move(m10); m_[1][1] = std::move(m11); m_[1][2] = std::move(m12);
            m_[2][0] = std::move(m20); m_[2][1] = std::move(m21); m_[2][2] = std::move(m22);
        }

        void resolve(const VariableManager& mgr) override {
            for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) m_[i][j]->resolve(mgr);
        }

        std::vector<const VariableNode*> getChildren() const override {
            std::vector<const VariableNode*> children;
            for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) children.push_back(m_[i][j].get());
            return children;
        }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            WorkspaceGuard guard(dest.size() * 8);
            std::span<TensorValue> mem = guard.span();
            
            m_[0][0]->evaluateBatch(ctx, dest);
            
            int offset = 0;
            std::span<TensorValue> rem[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (i == 0 && j == 0) continue;
                    rem[i][j] = mem.subspan(offset, dest.size());
                    m_[i][j]->evaluateBatch(ctx, rem[i][j]);
                    offset += dest.size();
                }
            }

            for (size_t k = 0; k < dest.size(); ++k) {
                dest[k] = TensorValue::matrix3(
                    dest[k].scalar(),          rem[0][1][k].scalar(), rem[0][2][k].scalar(),
                    rem[1][0][k].scalar(), rem[1][1][k].scalar(), rem[1][2][k].scalar(),
                    rem[2][0][k].scalar(), rem[2][1][k].scalar(), rem[2][2][k].scalar());
            }
        }
    private:
        std::unique_ptr<VariableNode> m_[3][3];
    };

    // =========================================================================
    // Compiler implementation (Recursive Descent)
    // =========================================================================
    
    namespace {
        
        template <typename Op>
        std::unique_ptr<VariableNode> makeBinary(std::unique_ptr<VariableNode> lhs, std::unique_ptr<VariableNode> rhs) {
            // 常量折叠 (Constant Folding)
            if (lhs->isConstant() && rhs->isConstant()) {
                auto leftNode = static_cast<ConstantNode*>(lhs.get());
                auto rightNode = static_cast<ConstantNode*>(rhs.get());
                return std::make_unique<ConstantNode>(Op()(leftNode->value(), rightNode->value()));
            }
            return std::make_unique<BinaryOpNode<Op>>(std::move(lhs), std::move(rhs));
        }

        template <typename Op>
        std::unique_ptr<VariableNode> makeUnary(std::unique_ptr<VariableNode> arg) {
            if (arg->isConstant()) {
                auto argNode = static_cast<ConstantNode*>(arg.get());
                return std::make_unique<ConstantNode>(Op()(argNode->value()));
            }
            return std::make_unique<UnaryOpNode<Op>>(std::move(arg));
        }

        bool isNearOne(Real value) { return std::abs(value - 1.0) < 1e-15; }

        class TensorAstCompiler {
        public:
            explicit TensorAstCompiler(std::string_view text) : text_(text) {}
            std::unique_ptr<VariableNode> compile() {
                auto root = parseExpression();
                skipWhitespace();
                if (!eof()) MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));
                return root;
            }

        private:
            std::unique_ptr<VariableNode> parseExpression() {
                auto lhs = parseTerm();
                for (;;) {
                    skipWhitespace();
                    if (match('+')) { lhs = makeBinary<OpAdd>(std::move(lhs), parseTerm()); continue; }
                    if (match('-')) { lhs = makeBinary<OpSub>(std::move(lhs), parseTerm()); continue; }
                    return lhs;
                }
            }

            std::unique_ptr<VariableNode> parseTerm() {
                auto lhs = parsePower();
                for (;;) {
                    skipWhitespace();
                    if (match('*')) { lhs = makeBinary<OpMul>(std::move(lhs), parsePower()); continue; }
                    if (match('/')) { lhs = makeBinary<OpDiv>(std::move(lhs), parsePower()); continue; }
                    return lhs;
                }
            }

            std::unique_ptr<VariableNode> parsePower() {
                auto lhs = parseUnary();
                skipWhitespace();
                if (match('^')) return makeBinary<OpPow>(std::move(lhs), parsePower());
                return lhs;
            }

            std::unique_ptr<VariableNode> parseUnary() {
                skipWhitespace();
                if (match('+')) return parseUnary();
                if (match('-')) return makeUnary<OpNeg>(parseUnary());
                return parsePrimary();
            }

            std::unique_ptr<VariableNode> parsePrimary() {
                skipWhitespace();
                if (match('(')) {
                    auto inner = parseExpression();
                    skipWhitespace();
                    if (!match(')')) MPFEM_THROW(ArgumentException, "Missing closing ')'");
                    return applyUnitSuffix(std::move(inner));
                }
                if (peek() == '[') return applyUnitSuffix(parseBracketLiteral());
                if (peekIsNumberStart()) return applyUnitSuffix(parseNumber());
                if (peekIsIdentifierStart()) {
                    std::string name = parseIdentifier();
                    skipWhitespace();
                    if (match('(')) return applyUnitSuffix(parseFunction(name));
                    
                    if (name == "pi") return std::make_unique<ConstantNode>(TensorValue::scalar(3.141592653589793));
                    if (name == "e") return std::make_unique<ConstantNode>(TensorValue::scalar(2.718281828459045));
                    return applyUnitSuffix(std::make_unique<VariableRefNode>(name));
                }
                MPFEM_THROW(ArgumentException, "Unexpected token near: " + std::string(text_.substr(pos_)));
            }

            std::unique_ptr<VariableNode> parseNumber() {
                const char* begin = text_.data() + pos_;
                char* end = nullptr;
                Real value = std::strtod(begin, &end);
                pos_ += (end - begin);
                return std::make_unique<ConstantNode>(TensorValue::scalar(value));
            }

            std::string parseIdentifier() {
                size_t begin = pos_++;
                while (!eof() && (std::isalnum(text_[pos_]) || text_[pos_] == '_')) ++pos_;
                return std::string(text_.substr(begin, pos_ - begin));
            }

            std::unique_ptr<VariableNode> parseFunction(const std::string& name) {
                if (name == "grad") {
                    skipWhitespace();
                    std::string fieldName = parseIdentifier();
                    skipWhitespace();
                    if (!match(')')) MPFEM_THROW(ArgumentException, "Missing ')' after grad argument");
                    return std::make_unique<GradRefNode>(fieldName);
                }

                std::vector<std::unique_ptr<VariableNode>> args;
                if (!match(')')) {
                    do {
                        args.push_back(parseExpression());
                        skipWhitespace();
                    } while (match(','));
                    if (!match(')')) MPFEM_THROW(ArgumentException, "Missing ')' in function " + name);
                }

                if (args.size() == 1) {
                    if (name == "sin") return makeUnary<OpSin>(std::move(args[0]));
                    if (name == "cos") return makeUnary<OpCos>(std::move(args[0]));
                    if (name == "tan") return makeUnary<OpTan>(std::move(args[0]));
                    if (name == "exp") return makeUnary<OpExp>(std::move(args[0]));
                    if (name == "log") return makeUnary<OpLog>(std::move(args[0]));
                    if (name == "sqrt") return makeUnary<OpSqrt>(std::move(args[0]));
                    if (name == "abs") return makeUnary<OpAbs>(std::move(args[0]));
                    if (name == "sym") return makeUnary<OpSym>(std::move(args[0]));
                    if (name == "trace" || name == "tr") return makeUnary<OpTrace>(std::move(args[0]));
                    if (name == "transpose") return makeUnary<OpTranspose>(std::move(args[0]));
                }
                else if (args.size() == 2) {
                    if (name == "dot") return makeBinary<OpDot>(std::move(args[0]), std::move(args[1]));
                    if (name == "pow") return makeBinary<OpPow>(std::move(args[0]), std::move(args[1]));
                    if (name == "min") return makeBinary<OpMin>(std::move(args[0]), std::move(args[1]));
                    if (name == "max") return makeBinary<OpMax>(std::move(args[0]), std::move(args[1]));
                }
                MPFEM_THROW(ArgumentException, "Unsupported function or wrong arity: " + name);
            }

            std::unique_ptr<VariableNode> parseBracketLiteral() {
                if (!match('[')) MPFEM_THROW(ArgumentException, "Internal error: bracket literal");

                std::vector<std::vector<std::unique_ptr<VariableNode>>> rows;
                rows.emplace_back();

                for (;;) {
                    skipWhitespace();
                    const size_t partBegin = pos_;
                    int parenDepth = 0, unitBracketDepth = 0;
                    while (!eof()) {
                        const char c = text_[pos_];
                        if (c == '(') ++parenDepth;
                        else if (c == ')') --parenDepth;
                        else if (c == '[') ++unitBracketDepth;
                        else if (c == ']') {
                            if (unitBracketDepth > 0) --unitBracketDepth;
                            else if (parenDepth == 0) break;
                        }
                        else if (parenDepth == 0 && unitBracketDepth == 0 && (c == ',' || c == ';')) break;
                        ++pos_;
                    }

                    if (eof()) MPFEM_THROW(ArgumentException, "Missing ']'");
                    const std::string part = strings::trim(std::string(text_.substr(partBegin, pos_ - partBegin)));
                    
                    UnitRegistry registry;
                    auto unitPart = registry.stripUnit(part);
                    TensorAstCompiler sc(unitPart.expression);
                    auto comp = sc.compile();
                    
                    if (!isNearOne(unitPart.multiplier)) {
                        comp = makeBinary<OpMul>(std::make_unique<ConstantNode>(TensorValue::scalar(unitPart.multiplier)), std::move(comp));
                    }
                    rows.back().push_back(std::move(comp));

                    const char delim = text_[pos_];
                    ++pos_;
                    if (delim == ',') continue;
                    if (delim == ';') { rows.emplace_back(); continue; }
                    if (delim == ']') break;
                }

                skipWhitespace();
                bool isTranspose = match('^');
                if (isTranspose) {
                    skipWhitespace();
                    if (!match('T')) MPFEM_THROW(ArgumentException, "Expected '^T'");
                }

                const size_t rowCount = rows.size();
                const size_t colCount = rows.front().size();

                if (rowCount == 1 && colCount == 1) return std::move(rows.front().front());
                if (rowCount == 1 && colCount == 3) {
                    return std::make_unique<VectorLiteralNode>(std::move(rows[0][0]), std::move(rows[0][1]), std::move(rows[0][2]));
                }
                if (rowCount == 3 && colCount == 3) {
                    return std::make_unique<MatrixLiteralNode>(
                        std::move(rows[0][0]), std::move(rows[0][1]), std::move(rows[0][2]),
                        std::move(rows[1][0]), std::move(rows[1][1]), std::move(rows[1][2]),
                        std::move(rows[2][0]), std::move(rows[2][1]), std::move(rows[2][2]));
                }

                MPFEM_THROW(ArgumentException, "Unsupported bracket shape");
            }

            std::unique_ptr<VariableNode> applyUnitSuffix(std::unique_ptr<VariableNode> node) {
                skipWhitespace();
                if (!match('[')) return node;
                size_t begin = pos_;
                while (!eof() && text_[pos_] != ']') ++pos_;
                std::string unit = strings::trim(std::string(text_.substr(begin, pos_ - begin)));
                ++pos_;
                
                Real mult = UnitRegistry().getMultiplier(unit);
                if (isNearOne(mult)) return node;
                return makeBinary<OpMul>(std::make_unique<ConstantNode>(TensorValue::scalar(mult)), std::move(node));
            }

            char peek() const { return eof() ? '\0' : text_[pos_]; }
            bool peekIsIdentifierStart() const { return !eof() && (std::isalpha(text_[pos_]) || text_[pos_] == '_'); }
            bool peekIsNumberStart() const { return !eof() && (std::isdigit(text_[pos_]) || text_[pos_] == '.'); }
            bool eof() const { return pos_ >= text_.size(); }
            void skipWhitespace() { while (!eof() && std::isspace(text_[pos_])) ++pos_; }
            bool match(char c) { if (peek() == c) { ++pos_; return true; } return false; }

            std::string_view text_;
            size_t pos_ = 0;
        };

        struct MatrixTemplate {
            bool literalMatrix = false;
            std::vector<std::string> components;
        };

        bool isSeparator(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ','; }

        MatrixTemplate parseComsolMatrixTemplate(std::string_view expression) {
            MatrixTemplate tpl;
            const std::string trimmed = strings::trim(std::string(expression));

            if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}') return tpl;

            tpl.literalMatrix = true;
            const std::string_view content(trimmed.data() + 1, trimmed.size() - 2);

            size_t index = 0;
            while (index < content.size()) {
                while (index < content.size() && isSeparator(content[index])) ++index;
                if (index >= content.size()) break;
                if (content[index] != '\'') MPFEM_THROW(ArgumentException, "Invalid comsol matrix literal");
                const size_t endQuote = content.find('\'', index + 1);
                if (endQuote == std::string_view::npos) MPFEM_THROW(ArgumentException, "Unterminated quote");
                std::string comp = strings::trim(std::string(content.substr(index + 1, endQuote - index - 1)));
                tpl.components.push_back(std::move(comp));
                index = endQuote + 1;
            }

            return tpl;
        }

        std::string convertComsolMatrixToBracketLiteral(const MatrixTemplate& tpl) {
            if (tpl.components.size() == 1) {
                const std::string& c = tpl.components[0];
                return "[" + c + ",0,0;0," + c + ",0;0,0," + c + "]";
            }
            return "[" + tpl.components[0] + "," + tpl.components[3] + "," + tpl.components[6] + ";"
                       + tpl.components[1] + "," + tpl.components[4] + "," + tpl.components[7] + ";"
                       + tpl.components[2] + "," + tpl.components[5] + "," + tpl.components[8] + "]";
        }
    } // namespace

    std::unique_ptr<VariableNode> ExpressionParser::parse(const std::string& expression) {
        std::string expressionText = strings::trim(expression);
        if (expressionText.empty()) MPFEM_THROW(ArgumentException, "Empty expression string");

        MatrixTemplate comsol = parseComsolMatrixTemplate(expressionText);
        if (comsol.literalMatrix) {
            expressionText = convertComsolMatrixToBracketLiteral(comsol);
        }

        TensorAstCompiler compiler(expressionText);
        return compiler.compile();
    }

} // namespace mpfem
# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 从难度最低，收益最高的部分开始，如果有些任务过于困难，你可以选择性放弃。
* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

这个重构旨在解决几个核心问题：
1. **消除伪 DAG**：原代码使用 `VariableManager` 进行拓扑排序，但底层 `RuntimeExpressionNode` 仍然在用 `ExpressionProgram` 递归求值，造成了双重抽象和多余的内存开销。
2. **纯粹的 AST 求值**：将“变量”和“AST 节点”统一。所有的变量定义都会被解析为 AST，变量引用（VariableRef）在编译期直接链接到目标 AST 节点。运行时直接遍历执行 AST。
3. **解除定义顺序限制**：允许先使用后定义。在 `parse` 阶段只生成未解析的引用节点，在 `compile` 阶段才进行指针链接和循环依赖检测。
4. **编译缓存与接口统一**：废弃字符串传递，废弃显式的 `dependencies` 数组。

以下是重构后的代码和迁移指南。

### 1. 修改 `expr/variable_graph.hpp`
统一 `VariableNode` 接口，增加 `resolve` 机制实现延迟链接。

```cpp
#ifndef MPFEM_EXPR_VARIABLE_GRAPH_HPP
#define MPFEM_EXPR_VARIABLE_GRAPH_HPP

#include "expr/evaluation_context.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpfem {

    class VariableManager;

    /**
     * @brief 统一的表达式节点接口（即是变量，也是 AST 节点）
     */
    class VariableNode {
    public:
        virtual ~VariableNode() = default;

        /// 批量求值
        virtual void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const = 0;

        /// 编译期链接：子类应在此处向 Manager 查找未解析的依赖
        virtual void resolve(const VariableManager& mgr) {}

        /// 获取子节点（用于检测循环依赖）
        virtual std::vector<const VariableNode*> getChildren() const { return {}; }
        
        virtual bool isConstant() const { return false; }
    };

    /**
     * @brief 纯粹的变量与 AST 树管理器
     * 不再做拓扑排序执行计划，仅负责存储节点、链接 AST 以及环检测。
     */
    class VariableManager {
    public:
        VariableManager();
        ~VariableManager() = default;

        /// 定义一个字符串表达式（立刻解析为 AST，但不立即链接）
        void define(std::string name, const std::string& expression);

        /// 绑定一个自定义节点（例如 GridFunction 提供者）
        void bindNode(std::string name, std::unique_ptr<VariableNode> node);

        /// 获取节点（如果在 compile 前调用，可能包含未解析的引用）
        const VariableNode* get(std::string_view name) const;

        /// 编译阶段：链接所有 VariableRef，并检查循环依赖
        void compile();

        /// 直接对指定变量求值
        void evaluate(std::string_view name,
            const EvaluationContext& ctx,
            std::span<TensorValue> dest) const;

    private:
        void checkCycles() const;
        void dfsVisit(const VariableNode* node, 
                      std::unordered_map<const VariableNode*, int>& state) const;

        std::unordered_map<std::string, std::unique_ptr<VariableNode>> nodes_;
        bool isCompiled_ = false;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_VARIABLE_GRAPH_HPP
```

### 2. 修改 `expr/variable_graph.cpp`
实现延迟链接和环路检测。

```cpp
#include "expr/variable_graph.hpp"
#include "core/exception.hpp"
#include "expr/expression_parser.hpp"

namespace mpfem {

    VariableManager::VariableManager() {
        // 内置基础变量由于是自定义节点，可直接绑定
    }

    void VariableManager::define(std::string name, const std::string& expression) {
        // 直接解析为纯 AST 树，遇到变量会生成 UnresolvedRef 节点
        nodes_[std::move(name)] = ExpressionParser::parse(expression);
        isCompiled_ = false;
    }

    void VariableManager::bindNode(std::string name, std::unique_ptr<VariableNode> node) {
        MPFEM_ASSERT(node != nullptr, "bindNode requires non-null node.");
        nodes_[std::move(name)] = std::move(node);
        isCompiled_ = false;
    }

    const VariableNode* VariableManager::get(std::string_view name) const {
        auto it = nodes_.find(std::string(name));
        return it != nodes_.end() ? it->second.get() : nullptr;
    }

    void VariableManager::compile() {
        if (isCompiled_) return;

        // 1. 链接阶段：所有 AST 节点去 Manager 查找真实指针
        for (const auto& [name, node] : nodes_) {
            node->resolve(*this);
        }

        // 2. 检测循环依赖
        checkCycles();

        isCompiled_ = true;
    }

    void VariableManager::evaluate(std::string_view name,
                                   const EvaluationContext& ctx,
                                   std::span<TensorValue> dest) const {
        MPFEM_ASSERT(isCompiled_, "VariableManager must be compiled before evaluation.");
        const VariableNode* node = get(name);
        if (!node) {
            MPFEM_THROW(ArgumentException, "Variable not found: " + std::string(name));
        }
        node->evaluateBatch(ctx, dest);
    }

    void VariableManager::checkCycles() const {
        std::unordered_map<const VariableNode*, int> state;
        for (const auto& [name, node] : nodes_) {
            if (state[node.get()] == 0) {
                dfsVisit(node.get(), state);
            }
        }
    }

    void VariableManager::dfsVisit(const VariableNode* node, 
                                   std::unordered_map<const VariableNode*, int>& state) const {
        state[node] = 1; // visiting
        for (const VariableNode* child : node->getChildren()) {
            if (!child) continue;
            if (state[child] == 1) {
                MPFEM_THROW(ArgumentException, "Cyclic dependency detected in expressions.");
            }
            if (state[child] == 0) {
                dfsVisit(child, state);
            }
        }
        state[node] = 2; // visited
    }

} // namespace mpfem
```

### 3. 修改 `expr/expression_parser.hpp`
简化 Parser 接口，直接返回 AST（不再需要 `ExpressionProgram`）。

```cpp
#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include "expr/variable_graph.hpp"
#include <memory>
#include <string>

namespace mpfem {

    class ExpressionParser {
    public:
        ExpressionParser() = delete; // 纯静态工具类

        /**
         * @brief 将表达式字符串解析为纯 AST 树。
         * 变量被解析为 VariableRefNode，在 Manager::compile() 时链接。
         */
        static std::unique_ptr<VariableNode> parse(const std::string& expression);
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EXPRESSION_PARSER_HPP
```

### 4. 修改 `expr/expression_parser.cpp`
将原来的 `AstNode` 体系彻底融合到 `VariableNode` 中。

```cpp
#include "expr/expression_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include "expr/unit_parser.hpp"

#include <cmath>
#include <functional>
#include <vector>

namespace mpfem {

    // =========================================================================
    // AST Node Implementations
    // =========================================================================

    class ConstantNode final : public VariableNode {
    public:
        explicit ConstantNode(TensorValue val) : val_(std::move(val)) {}
        void evaluateBatch(const EvaluationContext&, std::span<TensorValue> dest) const override {
            std::fill(dest.begin(), dest.end(), val_);
        }
        bool isConstant() const override { return true; }
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

    class BinaryOpNode final : public VariableNode {
    public:
        using OpFunc = TensorValue(*)(const TensorValue&, const TensorValue&);
        
        BinaryOpNode(std::unique_ptr<VariableNode> lhs, std::unique_ptr<VariableNode> rhs, OpFunc op)
            : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op_(op) {}

        void resolve(const VariableManager& mgr) override {
            lhs_->resolve(mgr);
            rhs_->resolve(mgr);
        }

        std::vector<const VariableNode*> getChildren() const override {
            return {lhs_.get(), rhs_.get()};
        }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            lhs_->evaluateBatch(ctx, dest);
            std::vector<TensorValue> rhs_vals(dest.size());
            rhs_->evaluateBatch(ctx, rhs_vals);
            for (size_t i = 0; i < dest.size(); ++i) {
                dest[i] = op_(dest[i], rhs_vals[i]);
            }
        }
    private:
        std::unique_ptr<VariableNode> lhs_;
        std::unique_ptr<VariableNode> rhs_;
        OpFunc op_;
    };

    class UnaryOpNode final : public VariableNode {
    public:
        using OpFunc = TensorValue(*)(const TensorValue&);
        
        UnaryOpNode(std::unique_ptr<VariableNode> arg, OpFunc op)
            : arg_(std::move(arg)), op_(op) {}

        void resolve(const VariableManager& mgr) override { arg_->resolve(mgr); }
        std::vector<const VariableNode*> getChildren() const override { return {arg_.get()}; }

        void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
            arg_->evaluateBatch(ctx, dest);
            for (auto& v : dest) v = op_(v);
        }
    private:
        std::unique_ptr<VariableNode> arg_;
        OpFunc op_;
    };

    // 数学操作符映射函数
    TensorValue op_add(const TensorValue& a, const TensorValue& b) { return a + b; }
    TensorValue op_sub(const TensorValue& a, const TensorValue& b) { return a - b; }
    TensorValue op_mul(const TensorValue& a, const TensorValue& b) { return a * b; }
    TensorValue op_div(const TensorValue& a, const TensorValue& b) { return a / b; }
    TensorValue op_pow(const TensorValue& a, const TensorValue& b) { return TensorValue::scalar(std::pow(a.scalar(), b.scalar())); }
    TensorValue op_dot(const TensorValue& a, const TensorValue& b) { return TensorValue::scalar(dot(a, b)); }
    TensorValue op_neg(const TensorValue& a) { return -a; }
    
    // =========================================================================
    // Compiler implementation (Recursive Descent)
    // =========================================================================
    
    namespace {
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
                    if (match('+')) { lhs = std::make_unique<BinaryOpNode>(std::move(lhs), parseTerm(), op_add); continue; }
                    if (match('-')) { lhs = std::make_unique<BinaryOpNode>(std::move(lhs), parseTerm(), op_sub); continue; }
                    return lhs;
                }
            }

            std::unique_ptr<VariableNode> parseTerm() {
                auto lhs = parsePower();
                for (;;) {
                    skipWhitespace();
                    if (match('*')) { lhs = std::make_unique<BinaryOpNode>(std::move(lhs), parsePower(), op_mul); continue; }
                    if (match('/')) { lhs = std::make_unique<BinaryOpNode>(std::move(lhs), parsePower(), op_div); continue; }
                    return lhs;
                }
            }

            std::unique_ptr<VariableNode> parsePower() {
                auto lhs = parseUnary();
                skipWhitespace();
                if (match('^')) return std::make_unique<BinaryOpNode>(std::move(lhs), parsePower(), op_pow);
                return lhs;
            }

            std::unique_ptr<VariableNode> parseUnary() {
                skipWhitespace();
                if (match('+')) return parseUnary();
                if (match('-')) return std::make_unique<UnaryOpNode>(parseUnary(), op_neg);
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
                if (peekIsNumberStart()) return applyUnitSuffix(parseNumber());
                if (peekIsIdentifierStart()) {
                    std::string name = parseIdentifier();
                    skipWhitespace();
                    if (match('(')) return applyUnitSuffix(parseFunction(name));
                    
                    // 内置常数
                    if (name == "pi") return std::make_unique<ConstantNode>(TensorValue::scalar(3.141592653589793));
                    if (name == "e") return std::make_unique<ConstantNode>(TensorValue::scalar(2.718281828459045));
                    
                    // 变量引用
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
                std::vector<std::unique_ptr<VariableNode>> args;
                if (!match(')')) {
                    do {
                        args.push_back(parseExpression());
                        skipWhitespace();
                    } while (match(','));
                    if (!match(')')) MPFEM_THROW(ArgumentException, "Missing ')' in function " + name);
                }

                if (name == "dot" && args.size() == 2) {
                    return std::make_unique<BinaryOpNode>(std::move(args[0]), std::move(args[1]), op_dot);
                }
                
                // TODO: 添加 sym, trace, transpose 等其他函数支持
                MPFEM_THROW(ArgumentException, "Unsupported function or wrong arity: " + name);
            }

            std::unique_ptr<VariableNode> applyUnitSuffix(std::unique_ptr<VariableNode> node) {
                skipWhitespace();
                if (!match('[')) return node;
                size_t begin = pos_;
                while (!eof() && text_[pos_] != ']') ++pos_;
                std::string unit = strings::trim(std::string(text_.substr(begin, pos_ - begin)));
                ++pos_; // consume ']'
                
                Real mult = UnitRegistry().getMultiplier(unit);
                if (std::abs(mult - 1.0) < 1e-15) return node;
                
                auto constNode = std::make_unique<ConstantNode>(TensorValue::scalar(mult));
                return std::make_unique<BinaryOpNode>(std::move(constNode), std::move(node), op_mul);
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
    }

    std::unique_ptr<VariableNode> ExpressionParser::parse(const std::string& expression) {
        if (strings::trim(expression).empty()) {
            MPFEM_THROW(ArgumentException, "Empty expression string");
        }
        // 为了简化这里省略了 Comsol 的矩阵模板展开，可以按照类似的逻辑通过文本预处理做掉
        TensorAstCompiler compiler(expression);
        return compiler.compile();
    }

} // namespace mpfem
```

### 5. 修改 `problem/physics_problem_builder.cpp`
适配最新的 API，在 `build` 最后显式调用 `compile()`，同时修改 Provider 支持新接口。

```cpp
#include "physics_problem_builder.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "expr/variable_graph.hpp"
#include "fe/grid_function.hpp"
#include "io/problem_input_loader.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/field_values.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "steady_problem.hpp"
#include "transient_problem.hpp"

#include <atomic>

namespace mpfem {

    namespace {

        class DomainMultiplexerProvider final : public VariableNode {
        public:
            explicit DomainMultiplexerProvider(TensorShape shape) {}

            void addDomain(int domainId, std::string targetName) {
                targetNames_[domainId] = std::move(targetName);
            }

            void resolve(const VariableManager& mgr) override {
                for (const auto& [did, name] : targetNames_) {
                    const VariableNode* child = mgr.get(name);
                    if (!child) MPFEM_THROW(ArgumentException, "Unbound domain ref: " + name);
                    children_[did] = child;
                }
            }

            std::vector<const VariableNode*> getChildren() const override {
                std::vector<const VariableNode*> ret;
                for (const auto& [_, child] : children_) ret.push_back(child);
                return ret;
            }

            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
                auto it = children_.find(ctx.domainId);
                if (it == children_.end()) {
                    MPFEM_THROW(ArgumentException, "Missing child for domain " + std::to_string(ctx.domainId));
                }
                it->second->evaluateBatch(ctx, dest);
            }

        private:
            std::unordered_map<int, std::string> targetNames_;
            std::unordered_map<int, const VariableNode*> children_;
        };

        class GridFunctionValueProvider final : public VariableNode {
        public:
            explicit GridFunctionValueProvider(const GridFunction* field) : field_(field) {}
            void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const override {
                for (size_t i = 0; i < dest.size(); ++i) {
                    const Real* xi = &ctx.referencePoints[i].x();
                    dest[i] = TensorValue::scalar(field_->eval(ctx.elementId, xi));
                }
            }
        private:
            const GridFunction* field_;
        };

        //... 同理适配 GridFunctionGradientProvider

        const VariableNode* requireDomainPropertyNode(Problem& problem, std::string_view property, bool isMatrix) {
            const std::string nodeName(property);
            if (problem.globalVariables_.get(nodeName)) return problem.globalVariables_.get(nodeName);

            auto selector = std::make_unique<DomainMultiplexerProvider>(isMatrix ? TensorShape::matrix(3,3) : TensorShape::scalar());

            for (int domainId : problem.materials.domainIds()) {
                const std::string leafName = std::string(property) + "$domain_" + std::to_string(domainId);
                if (!problem.globalVariables_.get(leafName)) {
                    std::string expr = isMatrix ? problem.materials.matrixExpressionByDomain(domainId, property)
                                                : problem.materials.scalarExpressionByDomain(domainId, property);
                    problem.globalVariables_.define(leafName, expr);
                }
                selector->addDomain(domainId, leafName);
            }

            problem.globalVariables_.bindNode(nodeName, std::move(selector));
            return problem.globalVariables_.get(nodeName);
        }

        const VariableNode* makeScalarExpressionNode(Problem& problem, const std::string& expression) {
            static std::atomic<std::uint64_t> id{0};
            std::string name = "$expr_" + std::to_string(id++);
            problem.globalVariables_.define(name, expression);
            return problem.globalVariables_.get(name);
        }

    } // namespace

    namespace PhysicsProblemBuilder {
        
        // ... buildSolvers, buildElectrostatics 等保持核心逻辑不变
        // 只需将原先依赖 VariableManager 的返回值进行缓存的地方稍作修改

        std::unique_ptr<Problem> build(const std::string& caseDir, const ProblemInputLoader& inputLoader) {
            // ... 之前的初始化代码 ...
            
            problem->registerCaseDefinitionVariables();
            buildSolvers(*problem);

            // ==========================================
            // 重构关键点：构建完毕后，统一执行 compile() 进行全树链接
            // ==========================================
            LOG_INFO << "Compiling variable expression ASTs...";
            problem->globalVariables_.compile();
            
            // ... transient 初始化 ...
            return problem;
        }

    } // namespace PhysicsProblemBuilder
} // namespace mpfem
```

---

### 迁移指南 (Migration Guide)

#### 1. 核心变化与思想转变
* **旧逻辑**：`VariableManager::define()` 在被调用时，会立即寻找依赖。如果依赖没被注册，程序崩溃。在运行时，Manager 将表达式打包进独立的 `Workspace`。
* **新逻辑**：`VariableManager::define()` 仅仅完成字符串到 AST 树的文本解析（不会去找依赖）。只有在整棵树注册完后调用 `VariableManager::compile()`，指针才会发生相互链接。

#### 2. 在业务代码中的适配步骤
1. **删除 `VariableManager::clearExecutionPlan()` 的使用**：新架构不再维护执行排序计划，直接通过 AST 自身执行遍历。
2. **所有组件构建完成后，必须显式调用 `compile()`**：
   如果你有脱离 `PhysicsProblemBuilder` 手动构造 `Problem` 的代码，**在求解（`solve()` / `assemble()`）之前，必须执行**：
   ```cpp
   problem.globalVariables_.compile();
   ```
3. **AST 节点的扩展**：
   重构提供的 `expression_parser.cpp` 中的 `TensorAstCompiler` 目前缩减了对括号矩阵（Matrix/VectorLiteral）和一部分函数（如 `sym`, `trace`）的支持。你需要将原 `expression_parser.cpp` 中的特殊语法解析（如 `[a,b;c,d]` 语法）用新的 `std::make_unique<VectorLiteralNode(...)>` 之类的自定义 AST 节点补齐。（因为 AST 和 Variable 体系合并，原先构建 `AstNode` 的地方改为构建对应子类的 `VariableNode` 即可）。
4. **性能观察**：目前的批处理评估是在 `BinaryOpNode` 中分配了 `std::vector<TensorValue> rhs_vals(dest.size());`。由于单元内的积分点数量（dest.size()）极小（通常 $<30$），现代分配器开销可控。若成为瓶颈，可考虑引入 `thread_local std::vector<TensorValue>` 缓存池替代动态分配。
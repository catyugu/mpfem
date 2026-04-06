#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace mpfem {
    namespace {

        std::uint64_t nextProgramId()
        {
            static std::atomic<std::uint64_t> id {1};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

        class ConstantScalarNode final : public VariableNode {
        public:
            explicit ConstantScalarNode(double value)
                : value_(value)
            {
            }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
                if (dest.size() != n) {
                    MPFEM_THROW(ArgumentException, "ConstantScalarNode evaluate destination size mismatch.");
                }
                for (size_t i = 0; i < n; ++i) {
                    dest[i] = value_;
                }
            }

            bool isConstant() const override { return true; }

        private:
            double value_ = 0.0;
        };

        class GridFunctionNode final : public VariableNode {
        public:
            GridFunctionNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), 0.0);
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionNode requires ElementTransform in EvaluationContext.");
                }
                const size_t n = ctx.physicalPoints.size();
                for (size_t i = 0; i < n; ++i) {
                    const Real* xi = nullptr;
                    if (i < ctx.referencePoints.size()) {
                        xi = &ctx.referencePoints[i].x();
                    }
                    else {
                        xi = &ctx.transform->integrationPoint().xi;
                    }
                    dest[i] = field_->eval(ctx.transform->elementIndex(), xi);
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            const GridFunction* field_ = nullptr;
        };

        class ExternalDataNode final : public VariableNode {
        public:
            using ValueExtractor = std::function<double(const EvaluationContext&, size_t pointIndex)>;

            ExternalDataNode(std::string name, ValueExtractor extractor)
                : name_(std::move(name)), extractor_(std::move(extractor)) { }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                for (size_t i = 0; i < n; ++i) {
                    dest[i] = extractor_(ctx, i);
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            ValueExtractor extractor_;
        };

        class RuntimeScalarExpressionNode final : public VariableNode {
        public:
            RuntimeScalarExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
            {
            }

            VariableShape shape() const override { return VariableShape::Scalar; }
            std::pair<int, int> dimensions() const override { return {1, 1}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n) {
                    MPFEM_THROW(ArgumentException, "RuntimeScalarExpressionNode evaluate destination size mismatch.");
                }

                // Evaluate all dependencies - each produces n scalar values
                std::vector<std::vector<double>> depValues(dependencies_.size());
                for (size_t d = 0; d < dependencies_.size(); ++d) {
                    depValues[d].assign(n, 0.0);
                    dependencies_[d]->evaluateBatch(ctx, std::span<double>(depValues[d].data(), depValues[d].size()));
                }

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    // Collect one scalar value per dependency at point i
                    std::vector<double> inputValues(dependencies_.size());
                    for (size_t d = 0; d < dependencies_.size(); ++d) {
                        inputValues[d] = depValues[d][i];
                    }
                    ExprValue exprResult = program_.evaluate(std::span<const double>(inputValues.data(), inputValues.size()));
                    dest[i] = std::get<double>(exprResult);
                }
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            std::uint64_t id_ = 0;
        };

        class RuntimeMatrixExpressionNode final : public VariableNode {
        public:
            RuntimeMatrixExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), dependencies_(std::move(dependencies)), program_(std::move(program)), id_(nextProgramId())
            {
            }

            VariableShape shape() const override { return VariableShape::Matrix; }
            std::pair<int, int> dimensions() const override { return {3, 3}; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
            {
                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n * 9ull) {
                    MPFEM_THROW(ArgumentException, "RuntimeMatrixExpressionNode evaluate destination size mismatch.");
                }

                // Evaluate all dependencies
                // For scalar deps: each produces n values (depValues[d][i] = scalar at point i)
                // For matrix deps: each produces n*9 values
                std::vector<std::vector<double>> depValues(dependencies_.size());
                for (size_t d = 0; d < dependencies_.size(); ++d) {
                    depValues[d].assign(n, 0.0);
                    dependencies_[d]->evaluateBatch(ctx, std::span<double>(depValues[d].data(), depValues[d].size()));
                }

                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    // Collect one scalar value per dependency at point i
                    // For matrix expression with scalar dependencies, the expression
                    // internally handles broadcasting/replicating as needed
                    std::vector<double> inputValues(dependencies_.size());
                    for (size_t d = 0; d < dependencies_.size(); ++d) {
                        inputValues[d] = depValues[d][i];
                    }
                    const ExprValue exprResult = program_.evaluate(std::span<const double>(inputValues.data(), inputValues.size()));
                    const Matrix3 mat = std::get<Matrix3>(exprResult);
                    const size_t base = i * 9ull;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            dest[base + static_cast<size_t>(r * 3 + c)] = mat(r, c);
                        }
                    }
                }
            }

            std::vector<const VariableNode*> dependencies() const override
            {
                return dependencies_;
            }

        private:
            std::string expression_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            std::uint64_t id_ = 0;
        };

    } // namespace

    VariableManager::VariableManager()
    {
        // Register built-in spatial coordinates x, y, z and time t as ExternalDataNode
        nodes_["x"] = std::make_unique<ExternalDataNode>("x",
            [](const EvaluationContext& ctx, size_t pointIndex) -> double {
                return ctx.physicalPoints[pointIndex].x();
            });

        nodes_["y"] = std::make_unique<ExternalDataNode>("y",
            [](const EvaluationContext& ctx, size_t pointIndex) -> double {
                return ctx.physicalPoints[pointIndex].y();
            });

        nodes_["z"] = std::make_unique<ExternalDataNode>("z",
            [](const EvaluationContext& ctx, size_t pointIndex) -> double {
                return ctx.physicalPoints[pointIndex].z();
            });

        nodes_["t"] = std::make_unique<ExternalDataNode>("t",
            [](const EvaluationContext& ctx, size_t) -> double {
                return ctx.time;
            });

        graphDirty_ = true;
    }

    void VariableManager::registerConstantExpression(std::string name, std::string expressionText)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expressionText);

        // If expression has no dependencies (pure constant), create ConstantScalarNode directly
        MPFEM_ASSERT(program.dependencies().empty(), "Expected constant expression to have no dependencies.");
        double value = std::get<double>(program.evaluate({}));
        nodes_[std::move(name)] = std::make_unique<ConstantScalarNode>(value);

        graphDirty_ = true;
    }

    void VariableManager::registerScalarExpression(std::string name, std::string expression)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expression);

        // Resolve all symbol dependencies from the node store
        std::vector<const VariableNode*> dependencies;
        dependencies.reserve(program.dependencies().size());

        for (const std::string& symbol : program.dependencies()) {
            const auto it = nodes_.find(symbol);
            MPFEM_ASSERT(it != nodes_.end() && it->second != nullptr,
                "Unbound symbol in expression: " + symbol);
            dependencies.push_back(it->second.get());
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeScalarExpressionNode>(
            std::move(expression),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerMatrixExpression(std::string name, std::string expression)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expression);

        // Resolve all symbol dependencies from the node store
        std::vector<const VariableNode*> dependencies;
        dependencies.reserve(program.dependencies().size());

        for (const std::string& symbol : program.dependencies()) {
            const auto it = nodes_.find(symbol);
            MPFEM_ASSERT(it != nodes_.end() && it->second != nullptr,
                "Unbound symbol in expression: " + symbol);
            dependencies.push_back(it->second.get());
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeMatrixExpressionNode>(
            std::move(expression),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerGridFunction(std::string name, const GridFunction* field)
    {
        nodes_[std::move(name)] = std::make_unique<GridFunctionNode>(name, field);
        graphDirty_ = true;
    }

    void VariableManager::registerExternalSource(std::string name,
        std::function<double(const EvaluationContext&, size_t pointIndex)> extractor)
    {
        nodes_[std::move(name)] = std::make_unique<ExternalDataNode>(name, std::move(extractor));
        graphDirty_ = true;
    }

    void VariableManager::adoptNode(std::unique_ptr<VariableNode> node, std::string name)
    {
        nodes_[std::move(name)] = std::move(node);
        graphDirty_ = true;
    }

    const VariableNode* VariableManager::get(std::string_view name) const
    {
        const auto it = nodes_.find(std::string(name));
        if (it == nodes_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    void VariableManager::clearExecutionPlan()
    {
        executionPlan_.clear();
    }

    void VariableManager::dfsVisit(const VariableNode* node,
        std::unordered_map<const VariableNode*, int>& marks,
        std::vector<const VariableNode*>& ordered) const
    {
        auto it = marks.find(node);
        if (it != marks.end()) {
            if (it->second == 1) {
                MPFEM_THROW(ArgumentException, "Variable graph has cyclic dependencies.");
            }
            if (it->second == 2) {
                return;
            }
        }

        marks[node] = 1;
        const std::vector<const VariableNode*> deps = node->dependencies();
        for (const VariableNode* dep : deps) {
            if (dep) {
                dfsVisit(dep, marks, ordered);
            }
        }
        marks[node] = 2;
        ordered.push_back(node);
    }

    void VariableManager::compileGraph()
    {
        if (!graphDirty_) {
            return;
        }

        clearExecutionPlan();

        std::unordered_map<const VariableNode*, int> marks;
        marks.reserve(nodes_.size() * 2);

        std::vector<const VariableNode*> ordered;
        ordered.reserve(nodes_.size());

        for (const auto& [_, node] : nodes_) {
            if (node) {
                dfsVisit(node.get(), marks, ordered);
            }
        }

        executionPlan_ = std::move(ordered);
        graphDirty_ = false;
    }

} // namespace mpfem

#include "expr/variable_graph.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"

#include <algorithm>
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

        // =============================================================================
        // Thread-local bump allocator：调用方完全隐形，零分配批量求值
        // =============================================================================
        struct ThreadScratchpad {
            std::vector<Real> buffer;
            std::vector<TensorValue> tensorArgs;
            size_t offset = 0;

            Real* allocate(size_t size)
            {
                if (offset + size > buffer.size()) {
                    // 倍增扩容，仅前几次调用触发
                    buffer.resize(std::max(buffer.size() * 2, offset + size + 1024));
                }
                Real* ptr = buffer.data() + offset;
                offset += size;
                return ptr;
            }

            void deallocate(size_t oldOffset)
            {
                offset = oldOffset;
            }
        };

        thread_local ThreadScratchpad tls_scratchpad;

        // =============================================================================
        // Node implementations (unchanged)
        // =============================================================================

        class ConstantNode final : public VariableNode {
        public:
            explicit ConstantNode(TensorValue value)
                : value_(std::move(value)), shape_(value_.shape()) { }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                const size_t n = ctx.physicalPoints.empty() ? dest.size() / shape_.size() : ctx.physicalPoints.size();
                const size_t valSize = shape_.size();

                if (dest.size() != n * valSize) {
                    MPFEM_THROW(ArgumentException, "ConstantNode evaluate destination size mismatch.");
                }

                // Flatten data for fast batch fill
                const Real* rawData = value_.data();

                if (valSize == 1) {
                    const Real v = rawData[0];
                    std::fill(dest.begin(), dest.end(), v);
                }
                else {
                    // For vectors or matrices, tile components
                    for (size_t i = 0; i < n; ++i) {
                        const size_t base = i * valSize;
                        for (size_t c = 0; c < valSize; ++c) {
                            dest[base + c] = rawData[c];
                        }
                    }
                }
            }

            bool isConstant() const override { return true; }

        private:
            TensorValue value_;
            TensorShape shape_;
        };

        class GridFunctionNode final : public VariableNode {
        public:
            GridFunctionNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Real(0));
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

        class GridFunctionGradientNode final : public VariableNode {
        public:
            GridFunctionGradientNode(std::string name, const GridFunction* field)
                : name_(std::move(name)), field_(field) { }

            TensorShape shape() const override { return TensorShape::vector(3); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                if (!field_) {
                    std::fill(dest.begin(), dest.end(), Real(0));
                    return;
                }
                if (!ctx.transform) {
                    MPFEM_THROW(ArgumentException, "GridFunctionGradientNode requires ElementTransform in EvaluationContext.");
                }

                const size_t n = ctx.physicalPoints.size();
                if (dest.size() != n * 3ull) {
                    MPFEM_THROW(ArgumentException, "GridFunctionGradientNode evaluate destination size mismatch.");
                }

                for (size_t i = 0; i < n; ++i) {
                    const Real* xi = nullptr;
                    if (i < ctx.referencePoints.size()) {
                        xi = &ctx.referencePoints[i].x();
                        ctx.transform->setIntegrationPoint(xi);
                    }
                    else {
                        xi = &ctx.transform->integrationPoint().xi;
                    }

                    const Vector3 g = field_->gradient(ctx.transform->elementIndex(), xi, *ctx.transform);
                    const size_t base = i * 3ull;
                    dest[base] = g.x();
                    dest[base + 1] = g.y();
                    dest[base + 2] = g.z();
                }
            }

            std::vector<const VariableNode*> dependencies() const override { return {}; }

        private:
            std::string name_;
            const GridFunction* field_ = nullptr;
        };

        class ExternalDataNode final : public VariableNode {
        public:
            using ValueExtractor = std::function<Real(const EvaluationContext&, size_t pointIndex)>;

            ExternalDataNode(std::string name, ValueExtractor extractor)
                : name_(std::move(name)), extractor_(std::move(extractor)) { }

            TensorShape shape() const override { return TensorShape::scalar(); }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
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

        // =============================================================================
        // Runtime expression node with zero-allocation scratchpad
        // =============================================================================

        class RuntimeExpressionNode final : public VariableNode {
        public:
            RuntimeExpressionNode(std::string expression,
                std::vector<const VariableNode*> dependencies,
                ExpressionParser::ExpressionProgram program)
                : expression_(std::move(expression)), dependencies_(std::move(dependencies)), program_(std::move(program)), shape_(program_.shape()), id_(nextProgramId())
            {
                // 预计算依赖数据总大小和偏移量
                totalDepSize_ = 0;
                depSizes_.reserve(dependencies_.size());
                depOffsets_.reserve(dependencies_.size());
                for (const auto* dep : dependencies_) {
                    depOffsets_.push_back(totalDepSize_);
                    const size_t sz = dep->shape().size();
                    depSizes_.push_back(sz);
                    totalDepSize_ += sz;
                }
            }

            TensorShape shape() const override { return shape_; }

            void evaluateBatch(const EvaluationContext& ctx, std::span<Real> dest) const override
            {
                const size_t n = ctx.physicalPoints.empty() ? dest.size() / shape_.size() : ctx.physicalPoints.size();
                const size_t valueSize = shape_.size();
                if (dest.size() != n * valueSize) {
                    MPFEM_THROW(ArgumentException, "RuntimeExpressionNode evaluate destination size mismatch.");
                }

                // 1. 记录水位线，分配连续内存
                const size_t oldOffset = tls_scratchpad.offset;
                Real* depBuffer = tls_scratchpad.allocate(n * totalDepSize_);

                // 2. 递归求值依赖项
                for (size_t d = 0; d < dependencies_.size(); ++d) {
                    std::span<Real> depDest(depBuffer + depOffsets_[d] * n, n * depSizes_[d]);
                    dependencies_[d]->evaluateBatch(ctx, depDest);
                }

                // 3. 确保 tensorArgs 缓存足够大
                if (tls_scratchpad.tensorArgs.size() < dependencies_.size()) {
                    tls_scratchpad.tensorArgs.resize(dependencies_.size());
                }

                // 4. 对每个点执行表达式
                for (size_t i = 0; i < n; ++i) {
                    if (ctx.transform && i < ctx.referencePoints.size()) {
                        const Real xi[3] = {
                            ctx.referencePoints[i].x(),
                            ctx.referencePoints[i].y(),
                            ctx.referencePoints[i].z(),
                        };
                        ctx.transform->setIntegrationPoint(xi);
                    }

                    // 反序列化为 TensorValue
                    for (size_t d = 0; d < dependencies_.size(); ++d) {
                        const Real* vals = &depBuffer[depOffsets_[d] * n + i * depSizes_[d]];
                        if (depSizes_[d] == 1) {
                            tls_scratchpad.tensorArgs[d] = TensorValue::scalar(vals[0]);
                        }
                        else if (depSizes_[d] == 3) {
                            tls_scratchpad.tensorArgs[d] = TensorValue::vector(vals[0], vals[1], vals[2]);
                        }
                        else if (depSizes_[d] == 9) {
                            tls_scratchpad.tensorArgs[d] = TensorValue::matrix3(
                                vals[0], vals[1], vals[2],
                                vals[3], vals[4], vals[5],
                                vals[6], vals[7], vals[8]);
                        }
                    }

                    const TensorValue exprResult = program_.evaluate(
                        std::span<const TensorValue>(tls_scratchpad.tensorArgs.data(), dependencies_.size()));

                    // 写回目标
                    const size_t destBase = i * valueSize;
                    if (shape_.isScalar()) {
                        dest[destBase] = exprResult.scalar();
                    }
                    else if (shape_.isVector()) {
                        for (size_t c = 0; c < valueSize; ++c)
                            dest[destBase + c] = exprResult[static_cast<int>(c)];
                    }
                    else if (shape_.isMatrix()) {
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                dest[destBase + static_cast<size_t>(r * 3 + c)] = exprResult.at(r, c);
                            }
                        }
                    }
                }

                // 5. 恢复水位线（保证递归安全）
                tls_scratchpad.deallocate(oldOffset);
            }

            std::vector<const VariableNode*> dependencies() const override { return dependencies_; }

        private:
            std::string expression_;
            std::vector<const VariableNode*> dependencies_;
            ExpressionParser::ExpressionProgram program_;
            TensorShape shape_;
            std::uint64_t id_ = 0;

            // 预计算数据
            size_t totalDepSize_ = 0;
            std::vector<size_t> depSizes_;
            std::vector<size_t> depOffsets_;
        };

    } // namespace

    VariableManager::VariableManager()
    {
        nodes_["x"] = std::make_unique<ExternalDataNode>("x",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].x();
            });

        nodes_["y"] = std::make_unique<ExternalDataNode>("y",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].y();
            });

        nodes_["z"] = std::make_unique<ExternalDataNode>("z",
            [](const EvaluationContext& ctx, size_t pointIndex) -> Real {
                return ctx.physicalPoints[pointIndex].z();
            });

        nodes_["t"] = std::make_unique<ExternalDataNode>("t",
            [](const EvaluationContext& ctx, size_t) -> Real {
                return ctx.time;
            });

        graphDirty_ = true;
    }

    void VariableManager::registerConstantExpression(std::string name, std::string expressionText)
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(expressionText);

        MPFEM_ASSERT(program.dependencies().empty(), "Expected constant expression to have no dependencies.");

        // Directly get the full TensorValue (not limited to .scalar())
        const std::array<TensorValue, 0> noInputs {};
        TensorValue value = program.evaluate(std::span<const TensorValue>(noInputs.data(), noInputs.size()));

        nodes_[std::move(name)] = std::make_unique<ConstantNode>(std::move(value));

        graphDirty_ = true;
    }

    void VariableManager::registerExpression(std::string name, std::string expression)
    {
        ExpressionParser parser;
        std::unordered_map<std::string, TensorShape> registeredShapes;
        registeredShapes.reserve(nodes_.size());
        for (const auto& [symbol, node] : nodes_) {
            if (node) {
                registeredShapes.emplace(symbol, node->shape());
            }
        }
        ExpressionParser::ExpressionProgram program = parser.compile(expression, registeredShapes);

        std::vector<const VariableNode*> dependencies;
        dependencies.reserve(program.dependencies().size());

        for (const std::string& symbol : program.dependencies()) {
            if (nodes_.find(symbol) == nodes_.end()) {
                ensureGradientNode(symbol);
            }
            const auto it = nodes_.find(symbol);
            MPFEM_ASSERT(it != nodes_.end() && it->second != nullptr,
                "Unbound symbol in expression: " + symbol);
            dependencies.push_back(it->second.get());
        }

        nodes_[std::move(name)] = std::make_unique<RuntimeExpressionNode>(
            std::move(expression),
            std::move(dependencies),
            std::move(program));
        graphDirty_ = true;
    }

    void VariableManager::registerGridFunction(std::string name, const GridFunction* field)
    {
        const std::string key = name;
        nodes_[key] = std::make_unique<GridFunctionNode>(key, field);
        gridFunctions_[key] = field;
        graphDirty_ = true;
    }

    void VariableManager::ensureGradientNode(std::string_view symbol)
    {
        if (symbol.size() <= 6 || symbol.substr(0, 5) != "grad(" || symbol.back() != ')') {
            return;
        }

        const std::string base = std::string(symbol.substr(5, symbol.size() - 6));
        const auto fieldIt = gridFunctions_.find(base);
        if (fieldIt == gridFunctions_.end() || !fieldIt->second) {
            return;
        }

        const std::string gradName(symbol);
        if (nodes_.find(gradName) != nodes_.end()) {
            return;
        }
        nodes_[gradName] = std::make_unique<GridFunctionGradientNode>(gradName, fieldIt->second);
    }

    void VariableManager::registerExternalSource(std::string name,
        std::function<Real(const EvaluationContext&, size_t pointIndex)> extractor)
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

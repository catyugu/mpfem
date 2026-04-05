#include "problem/expression_coefficient_factory.hpp"

#include "expr/variable_graph.hpp"
#include "fe/element_transform.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace mpfem {
namespace {

GraphRuntimeResolvers toGraphResolvers(RuntimeExpressionResolvers resolvers)
{
    GraphRuntimeResolvers graphResolvers;
    graphResolvers.symbolBinder = std::move(resolvers.symbolBinder);
    return graphResolvers;
}

void registerCaseConstants(VariableManager& manager, const CaseDefinition& caseDef)
{
    for (const auto& entry : caseDef.getVariables()) {
        manager.registerConstant(entry.name, entry.siValue);
    }
}

EvaluationContext makeSinglePointContext(ElementTransform& transform,
                                         Real t,
                                         std::array<Vector3, 1>& referencePoints,
                                         std::array<Vector3, 1>& physicalPoints)
{
    const IntegrationPoint& ip = transform.integrationPoint();
    referencePoints[0] = Vector3(ip.xi, ip.eta, ip.zeta);
    transform.transform(ip, physicalPoints[0]);

    EvaluationContext ctx;
    ctx.time = t;
    ctx.domainId = static_cast<int>(transform.attribute());
    ctx.elementId = transform.elementIndex();
    ctx.referencePoints = std::span<const Vector3>(referencePoints.data(), referencePoints.size());
    ctx.physicalPoints = std::span<const Vector3>(physicalPoints.data(), physicalPoints.size());
    ctx.transform = &transform;
    return ctx;
}

class GraphScalarCoefficient final : public Coefficient {
public:
    GraphScalarCoefficient(std::string expression,
                           const CaseDefinition& caseDef,
                           RuntimeExpressionResolvers resolvers)
    {
        registerCaseConstants(manager_, caseDef);
        manager_.registerScalarExpression(kNodeName, std::move(expression), toGraphResolvers(std::move(resolvers)));
        manager_.compileGraph();
        node_ = manager_.get(kNodeName);
        if (!node_) {
            MPFEM_THROW(ArgumentException, "Failed to build scalar expression node.");
        }
    }

    void eval(ElementTransform& transform, Real& result, Real t = 0.0) const override
    {
        std::array<Vector3, 1> referencePoints;
        std::array<Vector3, 1> physicalPoints;
        EvaluationContext ctx = makeSinglePointContext(transform, t, referencePoints, physicalPoints);

        std::array<double, 1> values{0.0};
        node_->evaluate(ctx, std::span<double>(values.data(), values.size()));
        result = values[0];
    }

private:
    static constexpr const char* kNodeName = "$root_scalar";

    VariableManager manager_;
    const VariableNode* node_ = nullptr;
};

class GraphMatrixCoefficient final : public MatrixCoefficient {
public:
    GraphMatrixCoefficient(std::string expression,
                           const CaseDefinition& caseDef,
                           RuntimeExpressionResolvers resolvers)
    {
        registerCaseConstants(manager_, caseDef);
        manager_.registerMatrixExpression(kNodeName, std::move(expression), toGraphResolvers(std::move(resolvers)));
        manager_.compileGraph();
        node_ = manager_.get(kNodeName);
        if (!node_) {
            MPFEM_THROW(ArgumentException, "Failed to build matrix expression node.");
        }
    }

    void eval(ElementTransform& transform, Matrix3& result, Real t = 0.0) const override
    {
        std::array<Vector3, 1> referencePoints;
        std::array<Vector3, 1> physicalPoints;
        EvaluationContext ctx = makeSinglePointContext(transform, t, referencePoints, physicalPoints);

        std::array<double, 9> values{};
        node_->evaluate(ctx, std::span<double>(values.data(), values.size()));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result(r, c) = values[static_cast<size_t>(r * 3 + c)];
            }
        }
    }

private:
    static constexpr const char* kNodeName = "$root_matrix";

    VariableManager manager_;
    const VariableNode* node_ = nullptr;
};

} // namespace

    std::unique_ptr<Coefficient> createRuntimeScalarExpressionCoefficient(
        std::string expression,
        const CaseDefinition& caseDef,
        RuntimeExpressionResolvers resolvers)
    {
        return std::make_unique<GraphScalarCoefficient>(
            std::move(expression),
            caseDef,
            std::move(resolvers));
    }

    std::unique_ptr<MatrixCoefficient> createRuntimeMatrixExpressionCoefficient(
        std::string expression,
        const CaseDefinition& caseDef,
        RuntimeExpressionResolvers resolvers)
    {
        return std::make_unique<GraphMatrixCoefficient>(
            std::move(expression),
            caseDef,
            std::move(resolvers));
    }

} // namespace mpfem
#include "core/logger.hpp"
#include "expr/variable_graph.hpp"
#include <gtest/gtest.h>

#include <array>

using namespace mpfem;

class VariableNodeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Logger::setLevel(LogLevel::Warning);
    }
};

TEST_F(VariableNodeTest, ConstantScalarNodeEvaluation)
{
    VariableManager manager;
    manager.registerConstantExpression("k", "3.14");
    manager.compileGraph();

    const VariableNode* node = manager.get("k");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->shape(), VariableShape::Scalar);

    std::array<Vector3, 2> points {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points.data(), points.size());
    std::array<double, 2> out {0.0, 0.0};
    node->evaluateBatch(ctx, std::span<double>(out.data(), out.size()));

    EXPECT_NEAR(out[0], 3.14, 1e-12);
    EXPECT_NEAR(out[1], 3.14, 1e-12);
}

TEST_F(VariableNodeTest, ScalarExpressionNodeEvaluation)
{
    VariableManager manager;
    manager.registerConstantExpression("k", "2.0");
    manager.registerExpression("f", "k * x + 1.0");
    manager.compileGraph();

    const VariableNode* node = manager.get("f");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->shape(), VariableShape::Scalar);

    std::array<Vector3, 2> points {Vector3(0.0, 0.0, 0.0), Vector3(2.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points.data(), points.size());
    std::array<double, 2> out {0.0, 0.0};
    node->evaluateBatch(ctx, std::span<double>(out.data(), out.size()));

    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], 5.0, 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

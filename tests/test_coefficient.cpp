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
    ASSERT_TRUE(node->shape().isScalar());

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
    ASSERT_TRUE(node->shape().isScalar());

    std::array<Vector3, 2> points {Vector3(0.0, 0.0, 0.0), Vector3(2.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points.data(), points.size());
    std::array<double, 2> out {0.0, 0.0};
    node->evaluateBatch(ctx, std::span<double>(out.data(), out.size()));

    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], 5.0, 1e-12);
}

TEST_F(VariableNodeTest, VectorLiteralMatMulAndDotEvaluation)
{
    VariableManager manager;
    manager.registerExpression("Iv", "[1,0,0;0,1,0;0,0,1] * [x,y,z]^T");
    manager.registerExpression("norm2", "dot([x,y,z]^T, [x,y,z]^T)");
    manager.compileGraph();

    const VariableNode* ivNode = manager.get("Iv");
    const VariableNode* norm2Node = manager.get("norm2");
    ASSERT_NE(ivNode, nullptr);
    ASSERT_NE(norm2Node, nullptr);
    ASSERT_TRUE(ivNode->shape().isVector());
    ASSERT_TRUE(norm2Node->shape().isScalar());

    std::array<Vector3, 2> points {Vector3(1.5, -2.0, 3.0), Vector3(0.5, 4.0, -1.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points.data(), points.size());

    std::array<double, 6> ivOut {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ivNode->evaluateBatch(ctx, std::span<double>(ivOut.data(), ivOut.size()));

    EXPECT_NEAR(ivOut[0], 1.5, 1e-12);
    EXPECT_NEAR(ivOut[1], -2.0, 1e-12);
    EXPECT_NEAR(ivOut[2], 3.0, 1e-12);
    EXPECT_NEAR(ivOut[3], 0.5, 1e-12);
    EXPECT_NEAR(ivOut[4], 4.0, 1e-12);
    EXPECT_NEAR(ivOut[5], -1.0, 1e-12);

    std::array<double, 2> norm2Out {0.0, 0.0};
    norm2Node->evaluateBatch(ctx, std::span<double>(norm2Out.data(), norm2Out.size()));

    EXPECT_NEAR(norm2Out[0], 1.5 * 1.5 + 4.0 + 9.0, 1e-12);
    EXPECT_NEAR(norm2Out[1], 0.25 + 16.0 + 1.0, 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

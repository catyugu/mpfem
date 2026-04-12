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
    manager.define("k", "3.14");
    manager.compile();

    const VariableNode* node = manager.get("k");
    ASSERT_NE(node, nullptr);
    std::array<Vector3, 2> points {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points);
    std::array<Tensor, 2> out {};
    node->evaluateBatch(ctx, std::span<Tensor>(out));

    ASSERT_TRUE(out[0].isScalar());
    EXPECT_NEAR(out[0].scalar(), 3.14, 1e-12);
    EXPECT_NEAR(out[1].scalar(), 3.14, 1e-12);
}

TEST_F(VariableNodeTest, ScalarExpressionNodeEvaluation)
{
    VariableManager manager;
    manager.define("k", "2.0");
    manager.define("f", "k * x + 1.0");
    manager.compile();

    const VariableNode* node = manager.get("f");
    ASSERT_NE(node, nullptr);
    std::array<Vector3, 2> points {Vector3(0.0, 0.0, 0.0), Vector3(2.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points);
    std::array<Tensor, 2> out {};
    node->evaluateBatch(ctx, std::span<Tensor>(out));

    ASSERT_TRUE(out[0].isScalar());
    EXPECT_NEAR(out[0].scalar(), 1.0, 1e-12);
    EXPECT_NEAR(out[1].scalar(), 5.0, 1e-12);
}

TEST_F(VariableNodeTest, VectorLiteralMatMulAndDotEvaluation)
{
    VariableManager manager;
    manager.define("Iv", "[1,0,0;0,1,0;0,0,1] * [x,y,z]^T");
    manager.define("norm2", "dot([x,y,z]^T, [x,y,z]^T)");
    manager.compile();

    const VariableNode* ivNode = manager.get("Iv");
    const VariableNode* norm2Node = manager.get("norm2");
    ASSERT_NE(ivNode, nullptr);
    ASSERT_NE(norm2Node, nullptr);
    std::array<Vector3, 2> points {Vector3(1.5, -2.0, 3.0), Vector3(0.5, 4.0, -1.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points);

    std::array<Tensor, 2> ivOut {};
    ivNode->evaluateBatch(ctx, std::span<Tensor>(ivOut));

    ASSERT_TRUE(ivOut[0].isVector());
    Vector3 iv0 = ivOut[0].vector();
    Vector3 iv1 = ivOut[1].vector();
    EXPECT_NEAR(iv0.x(), 1.5, 1e-12);
    EXPECT_NEAR(iv0.y(), -2.0, 1e-12);
    EXPECT_NEAR(iv0.z(), 3.0, 1e-12);
    EXPECT_NEAR(iv1.x(), 0.5, 1e-12);
    EXPECT_NEAR(iv1.y(), 4.0, 1e-12);
    EXPECT_NEAR(iv1.z(), -1.0, 1e-12);

    std::array<Tensor, 2> norm2Out {};
    norm2Node->evaluateBatch(ctx, std::span<Tensor>(norm2Out));

    ASSERT_TRUE(norm2Out[0].isScalar());
    EXPECT_NEAR(norm2Out[0].scalar(), 1.5 * 1.5 + 4.0 + 9.0, 1e-12);
    EXPECT_NEAR(norm2Out[1].scalar(), 0.25 + 16.0 + 1.0, 1e-12);
}

TEST_F(VariableNodeTest, TensorSymTraceTransposeEvaluation)
{
    VariableManager manager;
    manager.define("S", "sym([1,2,3;4,5,6;7,8,9])");
    manager.define("St", "transpose([1,2,3;4,5,6;7,8,9])");
    manager.define("trS", "trace(sym([1,2,3;4,5,6;7,8,9]))");
    manager.compile();

    const VariableNode* sNode = manager.get("S");
    const VariableNode* stNode = manager.get("St");
    const VariableNode* trNode = manager.get("trS");
    ASSERT_NE(sNode, nullptr);
    ASSERT_NE(stNode, nullptr);
    ASSERT_NE(trNode, nullptr);
    std::array<Vector3, 1> points {Vector3(0.0, 0.0, 0.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(points);

    std::array<Tensor, 1> sOut {};
    sNode->evaluateBatch(ctx, std::span<Tensor>(sOut));
    ASSERT_TRUE(sOut[0].isMatrix());
    Matrix3 s = sOut[0].matrix();
    EXPECT_NEAR(s(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(s(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(s(0, 2), 5.0, 1e-12);
    EXPECT_NEAR(s(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(s(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(s(1, 2), 7.0, 1e-12);
    EXPECT_NEAR(s(2, 0), 5.0, 1e-12);
    EXPECT_NEAR(s(2, 1), 7.0, 1e-12);
    EXPECT_NEAR(s(2, 2), 9.0, 1e-12);

    std::array<Tensor, 1> stOut {};
    stNode->evaluateBatch(ctx, std::span<Tensor>(stOut));
    ASSERT_TRUE(stOut[0].isMatrix());
    Matrix3 st = stOut[0].matrix();
    EXPECT_NEAR(st(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(st(0, 1), 4.0, 1e-12);
    EXPECT_NEAR(st(0, 2), 7.0, 1e-12);
    EXPECT_NEAR(st(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(st(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(st(1, 2), 8.0, 1e-12);
    EXPECT_NEAR(st(2, 0), 3.0, 1e-12);
    EXPECT_NEAR(st(2, 1), 6.0, 1e-12);
    EXPECT_NEAR(st(2, 2), 9.0, 1e-12);

    std::array<Tensor, 1> trOut {};
    trNode->evaluateBatch(ctx, std::span<Tensor>(trOut));
    ASSERT_TRUE(trOut[0].isScalar());
    EXPECT_NEAR(trOut[0].scalar(), 15.0, 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "core/logger.hpp"
#include "expr/variable_graph.hpp"
#include <gtest/gtest.h>

#include <array>

using namespace mpfem;

class BracketLiteralTest : public ::testing::Test {
protected:
    void SetUp() override { Logger::setLevel(LogLevel::Warning); }
};

// Test 1: Scalar bracket literal [5]
TEST_F(BracketLiteralTest, ScalarBracketLiteral)
{
    VariableManager m;
    m.define("s", "[5]");
    m.compile();
    const VariableNode* node = m.get("s");
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->shape().isScalar());

    std::array<Vector3, 1> pts {Vector3(0, 0, 0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(pts.data(), pts.size());
    std::array<TensorValue, 1> out {};
    node->evaluateBatch(ctx, std::span<TensorValue>(out.data(), out.size()));
    EXPECT_NEAR(out[0].scalar(), 5.0, 1e-12);
}

// Test 2: Vector bracket literal [1, 2, 3]
TEST_F(BracketLiteralTest, VectorBracketLiteral)
{
    VariableManager m;
    m.define("v", "[1, 2, 3]");
    m.compile();
    const VariableNode* node = m.get("v");
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->shape().isVector());
    EXPECT_EQ(node->shape().dimension(0), 3);

    std::array<Vector3, 1> pts {Vector3(0, 0, 0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(pts.data(), pts.size());
    std::array<TensorValue, 1> out {};
    node->evaluateBatch(ctx, std::span<TensorValue>(out.data(), out.size()));
    Vector3 v = out[0].toVector3();
    EXPECT_NEAR(v(0), 1.0, 1e-12);
    EXPECT_NEAR(v(1), 2.0, 1e-12);
    EXPECT_NEAR(v(2), 3.0, 1e-12);
}

// Test 3: Matrix bracket literal [1,2,3;4,5,6;7,8,9]
TEST_F(BracketLiteralTest, MatrixBracketLiteral)
{
    VariableManager m;
    m.define("M", "[1, 2, 3; 4, 5, 6; 7, 8, 9]");
    m.compile();
    const VariableNode* node = m.get("M");
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->shape().isMatrix());
    EXPECT_EQ(node->shape().rows(), 3);
    EXPECT_EQ(node->shape().cols(), 3);

    std::array<Vector3, 1> pts {Vector3(0, 0, 0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(pts.data(), pts.size());
    std::array<TensorValue, 1> out {};
    node->evaluateBatch(ctx, std::span<TensorValue>(out.data(), out.size()));
    Matrix3 M = out[0].toMatrix3();

    // Row 0
    EXPECT_NEAR(M(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(M(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(M(0, 2), 3.0, 1e-12);
    // Row 1
    EXPECT_NEAR(M(1, 0), 4.0, 1e-12);
    EXPECT_NEAR(M(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(M(1, 2), 6.0, 1e-12);
    // Row 2
    EXPECT_NEAR(M(2, 0), 7.0, 1e-12);
    EXPECT_NEAR(M(2, 1), 8.0, 1e-12);
    EXPECT_NEAR(M(2, 2), 9.0, 1e-12);
}

// Test 4: Identity matrix times vector [1,0,0;0,1,0;0,0,1] * [x,y,z]
TEST_F(BracketLiteralTest, MatrixVectorExpression)
{
    VariableManager m;
    m.define("Iv", "[1[S/m], 0, 0; 0, 1[S/m], 0; 0, 0, 1[S/m]] * [x, y, z]");
    m.compile();
    const VariableNode* node = m.get("Iv");
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->shape().isVector());
    EXPECT_EQ(node->shape().dimension(0), 3);

    std::array<Vector3, 1> pts {Vector3(1.0, 2.0, 3.0)};
    EvaluationContext ctx;
    ctx.physicalPoints = std::span<const Vector3>(pts.data(), pts.size());
    std::array<TensorValue, 1> out {};
    node->evaluateBatch(ctx, std::span<TensorValue>(out.data(), out.size()));
    Vector3 v = out[0].toVector3();
    // Identity matrix times [x,y,z] should give [x,y,z]
    EXPECT_NEAR(v(0), 1.0, 1e-12);
    EXPECT_NEAR(v(1), 2.0, 1e-12);
    EXPECT_NEAR(v(2), 3.0, 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

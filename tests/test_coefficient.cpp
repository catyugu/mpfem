#include <gtest/gtest.h>
#include "fe/coefficient.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class CoefficientTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
};

// FunctionCoefficient tests (using lambda)
TEST_F(CoefficientTest, FunctionCoefficient_Constant) {
    auto coef = constantCoefficient(3.14);
    
    // Simple test - verify construction
    EXPECT_NE(coef.get(), nullptr);
}

TEST_F(CoefficientTest, FunctionCoefficient_Function) {
    auto coef = std::make_unique<FunctionCoefficient>(
        [](ElementTransform&, Real& result, Real t) {
            result = 1.0 + t;
        });
    
    EXPECT_NE(coef.get(), nullptr);
}

// VectorCoefficient tests
TEST_F(CoefficientTest, VectorCoefficient_Constant) {
    auto coef = constantVectorCoefficient(1.0, 2.0, 3.0);
    EXPECT_NE(coef.get(), nullptr);
}


TEST_F(CoefficientTest, MatrixCoefficient_Full) {
    Matrix3 mat;
    mat << 1, 0, 0,
           0, 2, 0,
           0, 0, 3;
    auto coef = constantMatrixCoefficient(mat);
    EXPECT_NE(coef.get(), nullptr);
}

// Product coefficient functionality is now achieved via lambda composition
TEST_F(CoefficientTest, ProductCoefficient_LambdaComposition) {
    auto c1 = constantCoefficient(2.0);
    auto c2 = constantCoefficient(3.0);
    
    // Lambda-based product coefficient
    auto productCoef = std::make_unique<FunctionCoefficient>(
        [&c1, &c2](ElementTransform& trans, Real& result, Real t) {
            Real v1 = 0.0, v2 = 0.0;
            c1->eval(trans, v1, t);
            c2->eval(trans, v2, t);
            result = v1 * v2;
        });
    
    EXPECT_NE(c1.get(), nullptr);
    EXPECT_NE(c2.get(), nullptr);
    EXPECT_NE(productCoef.get(), nullptr);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

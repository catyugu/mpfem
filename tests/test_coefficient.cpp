#include <gtest/gtest.h>
#include "fe/coefficient.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "fe/fe_collection.hpp"
#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <cmath>

using namespace mpfem;

class CoefficientTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
};

// ConstantCoefficient tests
TEST_F(CoefficientTest, ConstantCoefficient_Value) {
    ConstantCoefficient coef(3.14);
    EXPECT_DOUBLE_EQ(coef.get(), 3.14);
}

TEST_F(CoefficientTest, ConstantCoefficient_Default) {
    ConstantCoefficient coef;
    EXPECT_DOUBLE_EQ(coef.get(), 1.0);
}

TEST_F(CoefficientTest, ConstantCoefficient_SetValue) {
    ConstantCoefficient coef(1.0);
    coef.set(2.5);
    EXPECT_DOUBLE_EQ(coef.get(), 2.5);
}

// PWConstCoefficient tests
TEST_F(CoefficientTest, PWConstCoefficient_Basic) {
    PWConstCoefficient coef(2);
    coef.set(1, 10.0);
    coef.set(2, 20.0);
    
    EXPECT_DOUBLE_EQ(coef.get(1), 10.0);
    EXPECT_DOUBLE_EQ(coef.get(2), 20.0);
}

// FunctionCoefficient tests
TEST_F(CoefficientTest, FunctionCoefficient_Basic) {
    FunctionCoefficient coef([](Real x, Real y, Real z, Real t) {
        return x + 2.0 * y + t;
    });
    
    // Simple test - just verify construction
    EXPECT_NE(&coef, nullptr);
}

// ProductCoefficient tests
TEST_F(CoefficientTest, ProductCoefficient_Basic) {
    ConstantCoefficient a(2.0);
    ConstantCoefficient b(3.0);
    ProductCoefficient prod(&a, &b);
    
    // Product should be 2*3=6
    ElementTransform trans;
    EXPECT_DOUBLE_EQ(prod.eval(trans), 6.0);
}

// ScaledCoefficient tests
TEST_F(CoefficientTest, ScaledCoefficient_Basic) {
    ConstantCoefficient base(4.0);
    ScaledCoefficient scaled(&base, 0.5);
    
    // Scaled should be 4*0.5=2
    ElementTransform trans;
    EXPECT_DOUBLE_EQ(scaled.eval(trans), 2.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
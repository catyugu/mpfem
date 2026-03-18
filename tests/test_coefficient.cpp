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

// FunctionCoefficient tests
TEST_F(CoefficientTest, FunctionCoefficient_Basic) {
    FunctionCoefficient coef([](Real x, Real y, Real z, Real t) {
        return x + 2.0 * y + t;
    });
    
    // Simple test - just verify construction
    EXPECT_NE(&coef, nullptr);
}

// DomainMappedCoefficient tests
TEST_F(CoefficientTest, DomainMappedCoefficient_SetAll) {
    DomainMappedCoefficient coef;
    ConstantCoefficient c(5.0);
    
    coef.setAll(&c);
    EXPECT_FALSE(coef.empty());
    EXPECT_EQ(coef.get(1), &c);
    EXPECT_EQ(coef.get(99), &c);  // Default for any domain
}

TEST_F(CoefficientTest, DomainMappedCoefficient_DomainSpecific) {
    DomainMappedCoefficient coef;
    ConstantCoefficient c1(10.0);
    ConstantCoefficient c2(20.0);
    
    coef.set(1, &c1);
    coef.set(2, &c2);
    
    EXPECT_EQ(coef.get(1), &c1);
    EXPECT_EQ(coef.get(2), &c2);
    EXPECT_EQ(coef.get(99), nullptr);  // No default set
}

TEST_F(CoefficientTest, DomainMappedCoefficient_BatchSet) {
    DomainMappedCoefficient coef;
    ConstantCoefficient c(15.0);
    
    coef.set({1, 2, 3}, &c);
    
    EXPECT_EQ(coef.get(1), &c);
    EXPECT_EQ(coef.get(2), &c);
    EXPECT_EQ(coef.get(3), &c);
}

TEST_F(CoefficientTest, DomainMappedCoefficient_Override) {
    DomainMappedCoefficient coef;
    ConstantCoefficient c1(10.0);
    ConstantCoefficient c2(20.0);
    
    coef.set(1, &c1);
    coef.set(1, &c2);  // Override
    
    EXPECT_EQ(coef.get(1), &c2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
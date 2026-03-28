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

// ScalarCoefficient tests (using lambda)
TEST_F(CoefficientTest, ScalarCoefficient_Constant) {
    auto coef = constantCoefficient(3.14);
    
    // Simple test - verify construction
    EXPECT_NE(coef.get(), nullptr);
}

TEST_F(CoefficientTest, ScalarCoefficient_Function) {
    auto coef = std::make_unique<ScalarCoefficient>(
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

// MatrixCoefficient tests
TEST_F(CoefficientTest, MatrixCoefficient_Diagonal) {
    auto coef = diagonalMatrixCoefficient(5.0);
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

// DomainMappedScalarCoefficient tests
TEST_F(CoefficientTest, DomainMappedScalarCoefficient_SetAll) {
    DomainMappedScalarCoefficient coef;
    auto c = constantCoefficient(5.0);
    
    coef.setAll(c.get());
    EXPECT_FALSE(coef.empty());
    EXPECT_EQ(coef.get(1), c.get());
    EXPECT_EQ(coef.get(99), c.get());  // Default for any domain
}

TEST_F(CoefficientTest, DomainMappedScalarCoefficient_DomainSpecific) {
    DomainMappedScalarCoefficient coef;
    auto c1 = constantCoefficient(10.0);
    auto c2 = constantCoefficient(20.0);
    
    coef.set(1, c1.get());
    coef.set(2, c2.get());
    
    EXPECT_EQ(coef.get(1), c1.get());
    EXPECT_EQ(coef.get(2), c2.get());
    EXPECT_EQ(coef.get(99), nullptr);  // No default set
}

TEST_F(CoefficientTest, DomainMappedScalarCoefficient_BatchSet) {
    DomainMappedScalarCoefficient coef;
    auto c = constantCoefficient(15.0);
    
    coef.set({1, 2, 3}, c.get());
    
    EXPECT_EQ(coef.get(1), c.get());
    EXPECT_EQ(coef.get(2), c.get());
    EXPECT_EQ(coef.get(3), c.get());
}

TEST_F(CoefficientTest, DomainMappedScalarCoefficient_Override) {
    DomainMappedScalarCoefficient coef;
    auto c1 = constantCoefficient(10.0);
    auto c2 = constantCoefficient(20.0);
    
    coef.set(1, c1.get());
    coef.set(1, c2.get());  // Override
    
    EXPECT_EQ(coef.get(1), c2.get());
}

// DomainMappedMatrixCoefficient tests
TEST_F(CoefficientTest, DomainMappedMatrixCoefficient_Basic) {
    DomainMappedMatrixCoefficient coef;
    auto m = diagonalMatrixCoefficient(5.0);
    
    coef.set(1, m.get());
    EXPECT_FALSE(coef.empty());
    EXPECT_EQ(coef.get(1), m.get());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

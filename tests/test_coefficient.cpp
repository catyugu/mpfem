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


TEST_F(CoefficientTest, MatrixCoefficient_Full) {
    Matrix3 mat;
    mat << 1, 0, 0,
           0, 2, 0,
           0, 0, 3;
    auto coef = constantMatrixCoefficient(mat);
    EXPECT_NE(coef.get(), nullptr);
}

TEST_F(CoefficientTest, ProductCoefficient_Create) {
    auto c1 = constantCoefficient(2.0);
    auto c2 = constantCoefficient(3.0);
    ProductCoefficient coef(c1.get(), c2.get());
    EXPECT_NE(c1.get(), nullptr);
    EXPECT_NE(c2.get(), nullptr);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

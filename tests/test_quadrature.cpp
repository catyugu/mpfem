#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include "fe/quadrature.hpp"

using namespace mpfem;

// =============================================================================
// 1D Gauss-Legendre Quadrature Tests
// =============================================================================

class GaussQuadratureTest : public ::testing::TestWithParam<int> {
protected:
    int order() const { return GetParam(); }
};

TEST_P(GaussQuadratureTest, WeightsSumToTwo) {
    // Sum of weights should equal 2 for [-1, 1] interval
    auto weights = gauss::getWeights(order());
    Real sum = 0.0;
    for (Real w : weights) {
        sum += w;
    }
    EXPECT_NEAR(sum, 2.0, 1e-12);
}

TEST_P(GaussQuadratureTest, IntegrateConstant) {
    // Integral of 1 over [-1, 1] = 2
    auto [xi, w] = gauss::get1D(order());
    Real integral = 0.0;
    for (size_t i = 0; i < xi.size(); ++i) {
        integral += w[i] * 1.0;
    }
    EXPECT_NEAR(integral, 2.0, 1e-12);
}

TEST_P(GaussQuadratureTest, IntegrateLinear) {
    // Integral of x over [-1, 1] = 0
    auto [xi, w] = gauss::get1D(order());
    Real integral = 0.0;
    for (size_t i = 0; i < xi.size(); ++i) {
        integral += w[i] * xi[i];
    }
    EXPECT_NEAR(integral, 0.0, 1e-12);
}

TEST_P(GaussQuadratureTest, IntegrateQuadratic) {
    // Integral of x^2 over [-1, 1] = 2/3
    auto [xi, w] = gauss::get1D(order());
    Real integral = 0.0;
    for (size_t i = 0; i < xi.size(); ++i) {
        integral += w[i] * xi[i] * xi[i];
    }
    // Need at least 2 points for exact integration
    if (order() >= 2) {
        EXPECT_NEAR(integral, 2.0 / 3.0, 1e-12);
    }
}

TEST_P(GaussQuadratureTest, IntegratePolynomialExactness) {
    // n-point Gauss rule integrates 2n-1 degree polynomials exactly
    int n = order();
    
    // Test polynomial of degree 2n-1: x^(2n-1)
    // Integral over [-1, 1] is 0 for odd powers
    if (n >= 1) {
        auto [xi, w] = gauss::get1D(n);
        
        // Test degree 2n-1 polynomial
        Real integral = 0.0;
        for (size_t i = 0; i < xi.size(); ++i) {
            Real val = 1.0;
            for (int p = 0; p < 2*n - 1; ++p) {
                val *= xi[i];
            }
            integral += w[i] * val;
        }
        // Odd polynomial integrates to 0
        EXPECT_NEAR(integral, 0.0, 1e-10);
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, GaussQuadratureTest, 
    ::testing::Range(1, 6));

// =============================================================================
// Segment Quadrature Tests
// =============================================================================

TEST(SegmentQuadratureTest, BasicProperties) {
    auto rule = quadrature::getSegment(2);
    EXPECT_EQ(rule.size(), 2);
}

TEST(SegmentQuadratureTest, IntegrateQuadraticExact) {
    auto rule = quadrature::getSegment(2);
    
    // f(x) = x^2, integral over [-1,1] = 2/3
    Real integral = 0.0;
    for (const auto& ip : rule) {
        Real x = ip.xi;
        integral += ip.weight * x * x;
    }
    EXPECT_NEAR(integral, 2.0 / 3.0, 1e-12);
}

// =============================================================================
// Triangle Quadrature Tests  
// =============================================================================

class TriangleQuadratureTest : public ::testing::TestWithParam<int> {
protected:
    int order() const { return GetParam(); }
};

TEST_P(TriangleQuadratureTest, WeightsSumToArea) {
    // Sum of weights should equal 0.5 (area of reference triangle)
    auto rule = quadrature::getTriangle(order());
    Real sum = 0.0;
    for (const auto& ip : rule) {
        sum += ip.weight;
    }
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST_P(TriangleQuadratureTest, PointsInTriangle) {
    // All points should be inside the reference triangle
    auto rule = quadrature::getTriangle(order());
    for (const auto& ip : rule) {
        Real xi = ip.xi;
        Real eta = ip.eta;
        EXPECT_GE(xi, 0.0);
        EXPECT_GE(eta, 0.0);
        EXPECT_LE(xi + eta, 1.0);
    }
}

TEST_P(TriangleQuadratureTest, IntegrateConstant) {
    auto rule = quadrature::getTriangle(order());
    Real integral = 0.0;
    for (const auto& ip : rule) {
        integral += ip.weight * 1.0;
    }
    EXPECT_NEAR(integral, 0.5, 1e-12);
}

TEST_P(TriangleQuadratureTest, IntegrateLinear) {
    // Integral of xi over reference triangle = 1/6
    auto rule = quadrature::getTriangle(order());
    Real integral = 0.0;
    for (const auto& ip : rule) {
        integral += ip.weight * ip.xi;
    }
    EXPECT_NEAR(integral, 1.0 / 6.0, 1e-12);
}

TEST_P(TriangleQuadratureTest, IntegrateQuadratic) {
    // Integral of xi^2 over reference triangle = 1/12
    // Need order >= 2 for exact integration
    if (order() >= 2) {
        auto rule = quadrature::getTriangle(order());
        Real integral = 0.0;
        for (const auto& ip : rule) {
            integral += ip.weight * ip.xi * ip.xi;
        }
        EXPECT_NEAR(integral, 1.0 / 12.0, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, TriangleQuadratureTest, 
    ::testing::Range(1, 5));

// =============================================================================
// Square Quadrature Tests
// =============================================================================

TEST(SquareQuadratureTest, BasicProperties) {
    auto rule = quadrature::getSquare(2);
    EXPECT_EQ(rule.size(), 4);  // 2x2
}

TEST(SquareQuadratureTest, WeightsSumToArea) {
    // Sum of weights should equal 4 (area of [-1,1]^2)
    auto rule = quadrature::getSquare(2);
    Real sum = 0.0;
    for (const auto& ip : rule) {
        sum += ip.weight;
    }
    EXPECT_NEAR(sum, 4.0, 1e-12);
}

TEST(SquareQuadratureTest, IntegrateBilinear) {
    // Integral of xi * eta over [-1,1]^2 = 0
    auto rule = quadrature::getSquare(2);
    Real integral = 0.0;
    for (const auto& ip : rule) {
        integral += ip.weight * ip.xi * ip.eta;
    }
    EXPECT_NEAR(integral, 0.0, 1e-12);
}

TEST(SquareQuadratureTest, TensorProductProperty) {
    // Verify tensor product structure
    auto rule = quadrature::getSquare(2);
    auto xi1d = gauss::getPoints(2);
    auto w1d = gauss::getWeights(2);
    
    int idx = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(rule[idx].xi, xi1d[i], 1e-12);
            EXPECT_NEAR(rule[idx].eta, xi1d[j], 1e-12);
            EXPECT_NEAR(rule[idx].weight, w1d[i] * w1d[j], 1e-12);
            ++idx;
        }
    }
}

// =============================================================================
// Tetrahedron Quadrature Tests
// =============================================================================

class TetrahedronQuadratureTest : public ::testing::TestWithParam<int> {
protected:
    int order() const { return GetParam(); }
};

TEST_P(TetrahedronQuadratureTest, WeightsSumToVolume) {
    // Sum of weights should equal 1/6 (volume of reference tetrahedron)
    auto rule = quadrature::getTetrahedron(order());
    Real sum = 0.0;
    for (const auto& ip : rule) {
        sum += ip.weight;
    }
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);
}

TEST_P(TetrahedronQuadratureTest, PointsInTetrahedron) {
    // All points should be inside the reference tetrahedron
    auto rule = quadrature::getTetrahedron(order());
    for (const auto& ip : rule) {
        Real xi = ip.xi;
        Real eta = ip.eta;
        Real zeta = ip.zeta;
        EXPECT_GE(xi, 0.0);
        EXPECT_GE(eta, 0.0);
        EXPECT_GE(zeta, 0.0);
        EXPECT_LE(xi + eta + zeta, 1.0);
    }
}

TEST_P(TetrahedronQuadratureTest, IntegrateConstant) {
    auto rule = quadrature::getTetrahedron(order());
    Real integral = 0.0;
    for (const auto& ip : rule) {
        integral += ip.weight * 1.0;
    }
    EXPECT_NEAR(integral, 1.0 / 6.0, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(Orders, TetrahedronQuadratureTest, 
    ::testing::Range(1, 4));

// =============================================================================
// Cube Quadrature Tests
// =============================================================================

TEST(CubeQuadratureTest, BasicProperties) {
    auto rule = quadrature::getCube(2);
    EXPECT_EQ(rule.size(), 8);  // 2x2x2
}

TEST(CubeQuadratureTest, WeightsSumToVolume) {
    // Sum of weights should equal 8 (volume of [-1,1]^3)
    auto rule = quadrature::getCube(2);
    Real sum = 0.0;
    for (const auto& ip : rule) {
        sum += ip.weight;
    }
    EXPECT_NEAR(sum, 8.0, 1e-12);
}

// =============================================================================
// Generic Quadrature Factory Tests
// =============================================================================

TEST(QuadratureFactoryTest, GetByGeometry) {
    // Test the generic get() function
    auto seg = quadrature::get(Geometry::Segment, 2);
    EXPECT_EQ(seg.size(), 2);
    
    auto tri = quadrature::get(Geometry::Triangle, 2);
    EXPECT_GT(tri.size(), 0);
    
    auto sq = quadrature::get(Geometry::Square, 2);
    EXPECT_EQ(sq.size(), 4);
    
    auto tet = quadrature::get(Geometry::Tetrahedron, 2);
    EXPECT_GT(tet.size(), 0);
    
    auto cube = quadrature::get(Geometry::Cube, 2);
    EXPECT_EQ(cube.size(), 8);
}

// =============================================================================
// Integration Tests - Known Integrals
// =============================================================================

TEST(IntegrationAccuracyTest, GaussOrder3Exactness) {
    // 3-point Gauss should integrate degree 5 polynomials exactly
    auto [xi, w] = gauss::get1D(3);
    
    // Test f(x) = x^5, integral = 0
    Real integral = 0.0;
    for (size_t i = 0; i < xi.size(); ++i) {
        integral += w[i] * std::pow(xi[i], 5);
    }
    EXPECT_NEAR(integral, 0.0, 1e-12);
    
    // Test f(x) = x^4, integral = 2/5
    integral = 0.0;
    for (size_t i = 0; i < xi.size(); ++i) {
        integral += w[i] * std::pow(xi[i], 4);
    }
    EXPECT_NEAR(integral, 2.0 / 5.0, 1e-12);
}

TEST(IntegrationAccuracyTest, TriangleOrder2Exactness) {
    // Order 2 should integrate quadratic polynomials exactly
    auto rule = quadrature::getTriangle(2);
    
    // Integral of xi*eta over reference triangle
    // ∫∫ xi*eta dA = 1/24
    Real integral = 0.0;
    for (const auto& ip : rule) {
        integral += ip.weight * ip.xi * ip.eta;
    }
    EXPECT_NEAR(integral, 1.0 / 24.0, 1e-12);
}

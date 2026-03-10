/**
 * @file test_fem.cpp
 * @brief Test finite element shape functions and quadrature
 */

#include <gtest/gtest.h>
#include "fem/fe_base.hpp"
#include "fem/fe_h1.hpp"
#include "fem/fe_collection.hpp"
#include "fem/quadrature.hpp"
#include "fem/element_transformation.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class FETest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().set_level(LogLevel::INFO);
    }
};

TEST_F(FETest, SegmentLinearShapeFunctions) {
    FE_Segment fe(1);

    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 2);
    EXPECT_EQ(fe.dim(), 1);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0, 0, 0), values);

    EXPECT_NEAR(values[0], 0.5, 1e-10);
    EXPECT_NEAR(values[1], 0.5, 1e-10);

    fe.shape_values(Point<3>(-1, 0, 0), values);
    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[1], 0.0, 1e-10);

    fe.shape_values(Point<3>(1, 0, 0), values);
    EXPECT_NEAR(values[0], 0.0, 1e-10);
    EXPECT_NEAR(values[1], 1.0, 1e-10);
}

TEST_F(FETest, TriangleLinearShapeFunctions) {
    FE_Triangle fe(1);

    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 3);
    EXPECT_EQ(fe.dim(), 2);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(1.0/3.0, 1.0/3.0, 0), values);

    EXPECT_NEAR(values[0], 1.0/3.0, 1e-10);
    EXPECT_NEAR(values[1], 1.0/3.0, 1e-10);
    EXPECT_NEAR(values[2], 1.0/3.0, 1e-10);

    fe.shape_values(Point<3>(0, 0, 0), values);
    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[1], 0.0, 1e-10);
    EXPECT_NEAR(values[2], 0.0, 1e-10);
}

TEST_F(FETest, QuadrilateralLinearShapeFunctions) {
    FE_Quadrilateral fe(1);

    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 4);
    EXPECT_EQ(fe.dim(), 2);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0, 0, 0), values);

    EXPECT_NEAR(values[0], 0.25, 1e-10);
    EXPECT_NEAR(values[1], 0.25, 1e-10);
    EXPECT_NEAR(values[2], 0.25, 1e-10);
    EXPECT_NEAR(values[3], 0.25, 1e-10);

    Scalar sum = 0;
    for (auto v : values) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(FETest, TetrahedronLinearShapeFunctions) {
    FE_Tetrahedron fe(1);

    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 4);
    EXPECT_EQ(fe.dim(), 3);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0.25, 0.25, 0.25), values);

    EXPECT_NEAR(values[0], 0.25, 1e-10);
    EXPECT_NEAR(values[1], 0.25, 1e-10);
    EXPECT_NEAR(values[2], 0.25, 1e-10);
    EXPECT_NEAR(values[3], 0.25, 1e-10);

    fe.shape_values(Point<3>(0, 0, 0), values);
    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[1], 0.0, 1e-10);
    EXPECT_NEAR(values[2], 0.0, 1e-10);
    EXPECT_NEAR(values[3], 0.0, 1e-10);
}

TEST_F(FETest, HexahedronLinearShapeFunctions) {
    FE_Hexahedron fe(1);

    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 8);
    EXPECT_EQ(fe.dim(), 3);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0, 0, 0), values);

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(values[i], 0.125, 1e-10);
    }

    Scalar sum = 0;
    for (auto v : values) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(FETest, SegmentQuadrature) {
    FE_Segment fe(1);

    Scalar integral = 0;
    for (int q = 0; q < fe.n_quadrature_points(); ++q) {
        const auto& qp = fe.quadrature_point(q);
        integral += qp.weight;
    }
    EXPECT_NEAR(integral, 2.0, 1e-10);
}

TEST_F(FETest, TriangleQuadrature) {
    FE_Triangle fe(1);

    Scalar integral = 0;
    for (int q = 0; q < fe.n_quadrature_points(); ++q) {
        const auto& qp = fe.quadrature_point(q);
        integral += qp.weight;
    }
    EXPECT_NEAR(integral, 0.5, 1e-10);
}

TEST_F(FETest, TetrahedronQuadrature) {
    FE_Tetrahedron fe(1);

    Scalar integral = 0;
    for (int q = 0; q < fe.n_quadrature_points(); ++q) {
        const auto& qp = fe.quadrature_point(q);
        integral += qp.weight;
    }
    EXPECT_NEAR(integral, 1.0/6.0, 1e-10);
}

TEST_F(FETest, QuadraticTriangleShapeFunctions) {
    FE_Triangle fe(2);

    EXPECT_EQ(fe.degree(), 2);
    EXPECT_EQ(fe.dofs_per_cell(), 6);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(1.0/3.0, 1.0/3.0, 0), values);

    EXPECT_NEAR(values[0], -1.0/9.0, 1e-10);
    EXPECT_NEAR(values[1], -1.0/9.0, 1e-10);
    EXPECT_NEAR(values[2], -1.0/9.0, 1e-10);
}

TEST_F(FETest, QuadraticTetrahedronShapeFunctions) {
    FE_Tetrahedron fe(2);

    EXPECT_EQ(fe.degree(), 2);
    EXPECT_EQ(fe.dofs_per_cell(), 10);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0.1, 0.2, 0.3), values);

    Scalar sum = 0;
    for (auto v : values) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(FETest, FECollectionFactory) {
    auto fe_tri = FECollection::create("Lagrange1", GeometryType::Triangle);
    EXPECT_EQ(fe_tri->dofs_per_cell(), 3);

    auto fe_tet = FECollection::create("Lagrange1", GeometryType::Tetrahedron);
    EXPECT_EQ(fe_tet->dofs_per_cell(), 4);

    auto fe_hex = FECollection::create("Lagrange1", GeometryType::Hexahedron);
    EXPECT_EQ(fe_hex->dofs_per_cell(), 8);
}

TEST_F(FETest, VectorElements) {
    FE_Tetrahedron fe(1, 3);

    EXPECT_EQ(fe.n_components(), 3);
    EXPECT_EQ(fe.dofs_per_cell(), 12);

    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0, 0, 0), values);

    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[4], 1.0, 1e-10);
    EXPECT_NEAR(values[8], 1.0, 1e-10);
}

TEST_F(FETest, ShapeGradients) {
    FE_Triangle fe(1);

    std::vector<Tensor<1, 3>> grads;
    fe.shape_gradients(Point<3>(0, 0, 0), grads);

    EXPECT_NEAR(grads[0].x(), -1.0, 1e-10);
    EXPECT_NEAR(grads[0].y(), -1.0, 1e-10);
    EXPECT_NEAR(grads[1].x(), 1.0, 1e-10);
    EXPECT_NEAR(grads[1].y(), 0.0, 1e-10);
    EXPECT_NEAR(grads[2].x(), 0.0, 1e-10);
    EXPECT_NEAR(grads[2].y(), 1.0, 1e-10);
}

TEST_F(FETest, PrecomputedShapeFunctions) {
    FE_Tetrahedron fe(1);

    for (int q = 0; q < fe.n_quadrature_points(); ++q) {
        std::vector<Scalar> values;
        fe.shape_values(fe.quadrature_point(q).coord, values);

        for (int i = 0; i < fe.dofs_per_cell(); ++i) {
            EXPECT_NEAR(fe.shape_value(i, q), values[i], 1e-10);
        }
    }
}

// Test wedge element shape functions
TEST_F(FETest, WedgeLinearShapeFunctions) {
    FE_Wedge fe(1);
    
    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 6);
    EXPECT_EQ(fe.dim(), 3);
    
    // Test at center of wedge (xi=1/3, eta=1/3, zeta=0)
    std::vector<Scalar> values;
    fe.shape_values(Point<3>(1.0/3.0, 1.0/3.0, 0), values);
    
    // All bottom nodes should have value 1/6, all top nodes 1/6
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(values[i], 1.0/6.0, 1e-10);
    }
    
    // Sum should be 1 (partition of unity)
    Scalar sum = 0;
    for (auto v : values) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
    
    // Test at bottom corner (0,0,-1)
    fe.shape_values(Point<3>(0, 0, -1), values);
    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[1], 0.0, 1e-10);
    
    // Test at top corner (0,0,1)
    fe.shape_values(Point<3>(0, 0, 1), values);
    EXPECT_NEAR(values[0], 0.0, 1e-10);
    EXPECT_NEAR(values[3], 1.0, 1e-10);
}

// Test pyramid element shape functions
TEST_F(FETest, PyramidLinearShapeFunctions) {
    FE_Pyramid fe(1);
    
    EXPECT_EQ(fe.degree(), 1);
    EXPECT_EQ(fe.dofs_per_cell(), 5);
    EXPECT_EQ(fe.dim(), 3);
    
    // Test at center of base (0,0,0)
    std::vector<Scalar> values;
    fe.shape_values(Point<3>(0, 0, 0), values);
    
    // At center of base, each base node has value 0.25
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(values[i], 0.25, 1e-10);
    }
    EXPECT_NEAR(values[4], 0.0, 1e-10);
    
    // Sum should be 1
    Scalar sum = 0;
    for (auto v : values) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
    
    // Test at apex (0,0,1)
    fe.shape_values(Point<3>(0, 0, 1), values);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(values[i], 0.0, 1e-10);
    }
    EXPECT_NEAR(values[4], 1.0, 1e-10);
    
    // Test at base corner (-1,-1,0)
    fe.shape_values(Point<3>(-1, -1, 0), values);
    EXPECT_NEAR(values[0], 1.0, 1e-10);
    EXPECT_NEAR(values[1], 0.0, 1e-10);
    EXPECT_NEAR(values[4], 0.0, 1e-10);
}

// Test wedge shape gradients
TEST_F(FETest, WedgeShapeGradients) {
    FE_Wedge fe(1);
    
    std::vector<Tensor<1, 3>> grads;
    fe.shape_gradients(Point<3>(0.3, 0.2, 0), grads);
    
    // Verify gradients are finite
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(grads[i].x()));
        EXPECT_TRUE(std::isfinite(grads[i].y()));
        EXPECT_TRUE(std::isfinite(grads[i].z()));
    }
}

// Test pyramid shape gradients
TEST_F(FETest, PyramidShapeGradients) {
    FE_Pyramid fe(1);
    
    std::vector<Tensor<1, 3>> grads;
    fe.shape_gradients(Point<3>(0, 0, 0), grads);
    
    // Verify gradients are finite
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(grads[i].x()));
        EXPECT_TRUE(std::isfinite(grads[i].y()));
        EXPECT_TRUE(std::isfinite(grads[i].z()));
    }
    
    // Test at apex (special case)
    fe.shape_gradients(Point<3>(0, 0, 1), grads);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(grads[i].x()));
        EXPECT_TRUE(std::isfinite(grads[i].y()));
        EXPECT_TRUE(std::isfinite(grads[i].z()));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
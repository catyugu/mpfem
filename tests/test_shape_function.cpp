#include <gtest/gtest.h>
#include <cmath>
#include "fe/shape_function.hpp"
#include "fe/quadrature.hpp"

using namespace mpfem;

// =============================================================================
// Segment Shape Function Tests
// =============================================================================

class SegmentShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        order_ = GetParam();
        shape_ = std::make_unique<H1SegmentShape>(order_);
    }
    
    int order_;
    std::unique_ptr<H1SegmentShape> shape_;
};

TEST_P(SegmentShapeTest, GeometryAndOrder) {
    EXPECT_EQ(shape_->geometry(), Geometry::Segment);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 1);
}

TEST_P(SegmentShapeTest, NumDofs) {
    // n dofs = order + 1
    EXPECT_EQ(shape_->numDofs(), order_ + 1);
}

TEST_P(SegmentShapeTest, PartitionOfUnity) {
    // Sum of shape functions = 1 at any point
    auto rule = quadrature::getSegment(std::max(1, order_));
    
    for (const auto& ip : rule) {
        auto values = shape_->evalValues(&ip.xi);
        Real sum = 0.0;
        for (Real v : values) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SegmentShapeTest, KroneckerDelta) {
    // Shape function i = 1 at dof point i
    auto coords = shape_->dofCoords();
    for (size_t i = 0; i < coords.size(); ++i) {
        auto values = shape_->evalValues(coords[i].data());
        for (size_t j = 0; j < values.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values[j], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(SegmentShapeTest, GradientSumZero) {
    // Sum of gradients = 0 (for partition of unity)
    auto rule = quadrature::getSegment(std::max(1, order_));
    
    for (const auto& ip : rule) {
        auto sv = shape_->eval(&ip.xi);
        Real sum = 0.0;
        for (const auto& grad : sv.gradients) {
            sum += grad.x();
        }
        EXPECT_NEAR(sum, 0.0, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, SegmentShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Triangle Shape Function Tests
// =============================================================================

class TriangleShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        order_ = GetParam();
        shape_ = std::make_unique<H1TriangleShape>(order_);
    }
    
    int order_;
    std::unique_ptr<H1TriangleShape> shape_;
};

TEST_P(TriangleShapeTest, GeometryAndOrder) {
    EXPECT_EQ(shape_->geometry(), Geometry::Triangle);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 2);
}

TEST_P(TriangleShapeTest, NumDofs) {
    // n dofs = (order+1)(order+2)/2
    int expected = (order_ + 1) * (order_ + 2) / 2;
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(TriangleShapeTest, PartitionOfUnity) {
    auto rule = quadrature::getTriangle(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta};
        auto values = shape_->evalValues(xi);
        Real sum = 0.0;
        for (Real v : values) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(TriangleShapeTest, KroneckerDeltaAtVertices) {
    // Test at vertices
    Real vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    
    for (int i = 0; i < 3; ++i) {
        auto values = shape_->evalValues(vertices[i]);
        EXPECT_NEAR(values[i], 1.0, 1e-12);
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TriangleShapeTest, GradientSumZero) {
    auto rule = quadrature::getTriangle(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta};
        auto sv = shape_->eval(xi);
        Real sum_x = 0.0, sum_y = 0.0;
        for (const auto& grad : sv.gradients) {
            sum_x += grad.x();
            sum_y += grad.y();
        }
        EXPECT_NEAR(sum_x, 0.0, 1e-12);
        EXPECT_NEAR(sum_y, 0.0, 1e-12);
    }
}

TEST_P(TriangleShapeTest, LinearGradientConstant) {
    // For linear triangle, gradients should be constant
    if (order_ == 1) {
        Real xi1[] = {0.1, 0.2};
        Real xi2[] = {0.3, 0.4};
        
        auto sv1 = shape_->eval(xi1);
        auto sv2 = shape_->eval(xi2);
        
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(sv1.gradients[i].x(), sv2.gradients[i].x(), 1e-12);
            EXPECT_NEAR(sv1.gradients[i].y(), sv2.gradients[i].y(), 1e-12);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, TriangleShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Square Shape Function Tests
// =============================================================================

class SquareShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        order_ = GetParam();
        shape_ = std::make_unique<H1SquareShape>(order_);
    }
    
    int order_;
    std::unique_ptr<H1SquareShape> shape_;
};

TEST_P(SquareShapeTest, GeometryAndOrder) {
    EXPECT_EQ(shape_->geometry(), Geometry::Square);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 2);
}

TEST_P(SquareShapeTest, NumDofs) {
    // n dofs = (order+1)^2
    int expected = (order_ + 1) * (order_ + 1);
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(SquareShapeTest, PartitionOfUnity) {
    auto rule = quadrature::getSquare(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta};
        auto values = shape_->evalValues(xi);
        Real sum = 0.0;
        for (Real v : values) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SquareShapeTest, TensorProductStructure) {
    // Square shape functions are tensor products of 1D functions
    H1SegmentShape seg(order_);
    
    Real xi[] = {0.3, -0.5};
    auto sq_values = shape_->evalValues(xi);
    auto seg_x = seg.evalValues(&xi[0]);
    auto seg_y = seg.evalValues(&xi[1]);
    
    int n = order_ + 1;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = j * n + i;
            Real expected = seg_x[i] * seg_y[j];
            EXPECT_NEAR(sq_values[idx], expected, 1e-12);
        }
    }
}

TEST_P(SquareShapeTest, KroneckerDeltaAtNodes) {
    auto coords = shape_->dofCoords();
    for (size_t i = 0; i < coords.size(); ++i) {
        auto values = shape_->evalValues(coords[i].data());
        for (size_t j = 0; j < values.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values[j], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(values[j], 0.0, 1e-12);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, SquareShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Tetrahedron Shape Function Tests
// =============================================================================

class TetrahedronShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        order_ = GetParam();
        shape_ = std::make_unique<H1TetrahedronShape>(order_);
    }
    
    int order_;
    std::unique_ptr<H1TetrahedronShape> shape_;
};

TEST_P(TetrahedronShapeTest, GeometryAndOrder) {
    EXPECT_EQ(shape_->geometry(), Geometry::Tetrahedron);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 3);
}

TEST_P(TetrahedronShapeTest, NumDofs) {
    // n dofs = (order+1)(order+2)(order+3)/6
    int expected = (order_ + 1) * (order_ + 2) * (order_ + 3) / 6;
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(TetrahedronShapeTest, PartitionOfUnity) {
    auto rule = quadrature::getTetrahedron(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta, ip.zeta};
        auto values = shape_->evalValues(xi);
        Real sum = 0.0;
        for (Real v : values) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(TetrahedronShapeTest, KroneckerDeltaAtVertices) {
    // Test at vertices
    Real vertices[4][3] = {
        {0.0, 0.0, 0.0},  // Vertex 0
        {1.0, 0.0, 0.0},  // Vertex 1
        {0.0, 1.0, 0.0},  // Vertex 2
        {0.0, 0.0, 1.0}   // Vertex 3
    };
    
    for (int i = 0; i < 4; ++i) {
        auto values = shape_->evalValues(vertices[i]);
        EXPECT_NEAR(values[i], 1.0, 1e-12);
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                EXPECT_NEAR(values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TetrahedronShapeTest, GradientSumZero) {
    auto rule = quadrature::getTetrahedron(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta, ip.zeta};
        auto sv = shape_->eval(xi);
        Real sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& grad : sv.gradients) {
            sum_x += grad.x();
            sum_y += grad.y();
            sum_z += grad.z();
        }
        EXPECT_NEAR(sum_x, 0.0, 1e-12);
        EXPECT_NEAR(sum_y, 0.0, 1e-12);
        EXPECT_NEAR(sum_z, 0.0, 1e-12);
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, TetrahedronShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Cube Shape Function Tests
// =============================================================================

class CubeShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        order_ = GetParam();
        shape_ = std::make_unique<H1CubeShape>(order_);
    }
    
    int order_;
    std::unique_ptr<H1CubeShape> shape_;
};

TEST_P(CubeShapeTest, GeometryAndOrder) {
    EXPECT_EQ(shape_->geometry(), Geometry::Cube);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 3);
}

TEST_P(CubeShapeTest, NumDofs) {
    // n dofs = (order+1)^3
    int expected = (order_ + 1) * (order_ + 1) * (order_ + 1);
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(CubeShapeTest, PartitionOfUnity) {
    auto rule = quadrature::getCube(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta, ip.zeta};
        auto values = shape_->evalValues(xi);
        Real sum = 0.0;
        for (Real v : values) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(CubeShapeTest, TensorProductStructure) {
    // Cube shape functions are tensor products of 1D functions
    H1SegmentShape seg(order_);
    
    Real xi[] = {0.2, -0.3, 0.4};
    auto cube_values = shape_->evalValues(xi);
    auto seg_x = seg.evalValues(&xi[0]);
    auto seg_y = seg.evalValues(&xi[1]);
    auto seg_z = seg.evalValues(&xi[2]);
    
    int n = order_ + 1;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = k * n * n + j * n + i;
                Real expected = seg_x[i] * seg_y[j] * seg_z[k];
                EXPECT_NEAR(cube_values[idx], expected, 1e-12);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, CubeShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Linear Element Specific Tests
// =============================================================================

TEST(LinearElementsTest, TriangleGradientAccuracy) {
    // For a linear triangle, gradients are constant and known
    H1TriangleShape shape(1);
    
    // At any point in the triangle
    Real xi[] = {0.2, 0.3};
    auto sv = shape.eval(xi);
    
    // φ0 = 1 - ξ - η, grad = (-1, -1)
    // φ1 = ξ, grad = (1, 0)
    // φ2 = η, grad = (0, 1)
    
    EXPECT_NEAR(sv.gradients[0].x(), -1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[0].y(), -1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[1].x(), 1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[1].y(), 0.0, 1e-12);
    EXPECT_NEAR(sv.gradients[2].x(), 0.0, 1e-12);
    EXPECT_NEAR(sv.gradients[2].y(), 1.0, 1e-12);
}

TEST(LinearElementsTest, TetrahedronGradientAccuracy) {
    // For a linear tetrahedron, gradients are constant
    H1TetrahedronShape shape(1);
    
    Real xi[] = {0.1, 0.2, 0.3};
    auto sv = shape.eval(xi);
    
    // φ0 = 1 - ξ - η - ζ, grad = (-1, -1, -1)
    // φ1 = ξ, grad = (1, 0, 0)
    // φ2 = η, grad = (0, 1, 0)
    // φ3 = ζ, grad = (0, 0, 1)
    
    EXPECT_NEAR(sv.gradients[0].x(), -1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[0].y(), -1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[0].z(), -1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[1].x(), 1.0, 1e-12);
    EXPECT_NEAR(sv.gradients[1].y(), 0.0, 1e-12);
    EXPECT_NEAR(sv.gradients[1].z(), 0.0, 1e-12);
}

// =============================================================================
// Quadratic Element Specific Tests
// =============================================================================

TEST(QuadraticElementsTest, TriangleQuadraticDofs) {
    H1TriangleShape shape(2);
    EXPECT_EQ(shape.numDofs(), 6);  // 3 vertices + 3 edges
    
    auto coords = shape.dofCoords();
    
    // Check vertex nodes
    EXPECT_NEAR(coords[0][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[0][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[1][0], 1.0, 1e-12);
    EXPECT_NEAR(coords[1][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[2][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[2][1], 1.0, 1e-12);
    
    // Check edge nodes (midpoints)
    EXPECT_NEAR(coords[3][0], 0.5, 1e-12);  // Edge 0-1
    EXPECT_NEAR(coords[3][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[4][0], 0.5, 1e-12);  // Edge 1-2
    EXPECT_NEAR(coords[4][1], 0.5, 1e-12);
    EXPECT_NEAR(coords[5][0], 0.0, 1e-12);  // Edge 2-0
    EXPECT_NEAR(coords[5][1], 0.5, 1e-12);
}

TEST(QuadraticElementsTest, TetrahedronQuadraticDofs) {
    H1TetrahedronShape shape(2);
    EXPECT_EQ(shape.numDofs(), 10);  // 4 vertices + 6 edges
    
    auto coords = shape.dofCoords();
    
    // Check vertex nodes (reference tetrahedron)
    // Vertex 0: (0, 0, 0)
    // Vertex 1: (1, 0, 0)
    // Vertex 2: (0, 1, 0)
    // Vertex 3: (0, 0, 1)
    EXPECT_NEAR(coords[0][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[0][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[0][2], 0.0, 1e-12);
    
    EXPECT_NEAR(coords[1][0], 1.0, 1e-12);
    EXPECT_NEAR(coords[1][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[1][2], 0.0, 1e-12);
    
    EXPECT_NEAR(coords[2][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[2][1], 1.0, 1e-12);
    EXPECT_NEAR(coords[2][2], 0.0, 1e-12);
    
    EXPECT_NEAR(coords[3][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[3][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[3][2], 1.0, 1e-12);
    
    // Check edge nodes (midpoints)
    // Edge 0-1: (0.5, 0, 0)
    EXPECT_NEAR(coords[4][0], 0.5, 1e-12);
    EXPECT_NEAR(coords[4][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[4][2], 0.0, 1e-12);
}

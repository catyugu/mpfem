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
        values_.resize(shape_->numDofs());
        grads_.resize(shape_->numDofs());
    }
    
    int order_;
    std::unique_ptr<H1SegmentShape> shape_;
    std::vector<Real> values_;
    std::vector<Vector3> grads_;
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
        shape_->evalValues(&ip.xi, values_.data());
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SegmentShapeTest, KroneckerDelta) {
    // Shape function i = 1 at dof point i
    auto coords = shape_->dofCoords();
    for (size_t i = 0; i < coords.size(); ++i) {
        shape_->evalValues(coords[i].data(), values_.data());
        for (size_t j = 0; j < values_.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values_[j], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(SegmentShapeTest, GradientSumZero) {
    // Sum of gradients = 0 (for partition of unity)
    auto rule = quadrature::getSegment(std::max(1, order_));
    
    for (const auto& ip : rule) {
        shape_->evalGrads(&ip.xi, grads_.data());
        Real sum = 0.0;
        for (const auto& grad : grads_) {
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
        values_.resize(shape_->numDofs());
        grads_.resize(shape_->numDofs());
    }
    
    int order_;
    std::unique_ptr<H1TriangleShape> shape_;
    std::vector<Real> values_;
    std::vector<Vector3> grads_;
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
        shape_->evalValues(xi, values_.data());
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(TriangleShapeTest, KroneckerDeltaAtVertices) {
    // Test at vertices
    Real vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    
    for (int i = 0; i < 3; ++i) {
        shape_->evalValues(vertices[i], values_.data());
        EXPECT_NEAR(values_[i], 1.0, 1e-12);
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TriangleShapeTest, GradientSumZero) {
    auto rule = quadrature::getTriangle(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta};
        shape_->evalGrads(xi, grads_.data());
        Real sum_x = 0.0, sum_y = 0.0;
        for (const auto& grad : grads_) {
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
        
        std::vector<Vector3> grads1(3), grads2(3);
        shape_->evalGrads(xi1, grads1.data());
        shape_->evalGrads(xi2, grads2.data());
        
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(grads1[i].x(), grads2[i].x(), 1e-12);
            EXPECT_NEAR(grads1[i].y(), grads2[i].y(), 1e-12);
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
        values_.resize(shape_->numDofs());
        grads_.resize(shape_->numDofs());
    }
    
    int order_;
    std::unique_ptr<H1SquareShape> shape_;
    std::vector<Real> values_;
    std::vector<Vector3> grads_;
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
        shape_->evalValues(xi, values_.data());
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SquareShapeTest, TensorProductStructure) {
    // Square shape functions use geometric node ordering (counter-clockwise)
    // not pure tensor product ordering (j*n+i)
    // For order 1: nodes are (-1,-1), (1,-1), (1,1), (-1,1)
    
    if (order_ == 2) {
        // For order 2, test that shape functions have correct values at nodes
        auto coords = shape_->dofCoords();
        for (size_t i = 0; i < coords.size(); ++i) {
            shape_->evalValues(coords[i].data(), values_.data());
            for (size_t j = 0; j < values_.size(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(values_[j], 1.0, 1e-12);
                } else {
                    EXPECT_NEAR(values_[j], 0.0, 1e-12);
                }
            }
        }
        return;
    }
    
    // For order 1, verify each node has the correct shape function value
    // based on geometric ordering
    H1SegmentShape seg(order_);
    std::vector<Real> seg_x(order_ + 1), seg_y(order_ + 1);
    
    Real xi[] = {0.3, -0.5};
    shape_->evalValues(xi, values_.data());
    seg.evalValues(&xi[0], seg_x.data());
    seg.evalValues(&xi[1], seg_y.data());
    
    // Geometric ordering: (-1,-1), (1,-1), (1,1), (-1,1)
    // seg_x[0] = phi at x=-1, seg_x[1] = phi at x=1
    // seg_y[0] = phi at y=-1, seg_y[1] = phi at y=1
    EXPECT_NEAR(values_[0], seg_x[0] * seg_y[0], 1e-12);  // (-1,-1)
    EXPECT_NEAR(values_[1], seg_x[1] * seg_y[0], 1e-12);  // ( 1,-1)
    EXPECT_NEAR(values_[2], seg_x[1] * seg_y[1], 1e-12);  // ( 1, 1)
    EXPECT_NEAR(values_[3], seg_x[0] * seg_y[1], 1e-12);  // (-1, 1)
}

TEST_P(SquareShapeTest, KroneckerDeltaAtNodes) {
    auto coords = shape_->dofCoords();
    for (size_t i = 0; i < coords.size(); ++i) {
        shape_->evalValues(coords[i].data(), values_.data());
        for (size_t j = 0; j < values_.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values_[j], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
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
        values_.resize(shape_->numDofs());
        grads_.resize(shape_->numDofs());
    }
    
    int order_;
    std::unique_ptr<H1TetrahedronShape> shape_;
    std::vector<Real> values_;
    std::vector<Vector3> grads_;
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
        shape_->evalValues(xi, values_.data());
        Real sum = 0.0;
        for (Real v : values_) {
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
        shape_->evalValues(vertices[i], values_.data());
        EXPECT_NEAR(values_[i], 1.0, 1e-12);
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TetrahedronShapeTest, GradientSumZero) {
    auto rule = quadrature::getTetrahedron(std::max(1, order_));
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta, ip.zeta};
        shape_->evalGrads(xi, grads_.data());
        Real sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& grad : grads_) {
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
        values_.resize(shape_->numDofs());
        grads_.resize(shape_->numDofs());
    }
    
    int order_;
    std::unique_ptr<H1CubeShape> shape_;
    std::vector<Real> values_;
    std::vector<Vector3> grads_;
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
        shape_->evalValues(xi, values_.data());
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(CubeShapeTest, TensorProductStructure) {
    // Cube shape functions use geometric node ordering, not pure tensor product ordering
    // For order 1: 8 corners arranged geometrically
    
    if (order_ == 2) {
        // For order 2, test that shape functions have correct values at nodes
        auto coords = shape_->dofCoords();
        for (size_t i = 0; i < coords.size(); ++i) {
            shape_->evalValues(coords[i].data(), values_.data());
            for (size_t j = 0; j < values_.size(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(values_[j], 1.0, 1e-12);
                } else {
                    EXPECT_NEAR(values_[j], 0.0, 1e-12);
                }
            }
        }
        return;
    }
    
    // For order 1, verify the geometric node ordering is correct
    // Node ordering: corners in geometric order
    // seg_x[0] = phi at x=-1, seg_x[1] = phi at x=1
    H1SegmentShape seg(order_);
    std::vector<Real> seg_x(order_ + 1), seg_y(order_ + 1), seg_z(order_ + 1);
    
    Real xi[] = {0.2, -0.3, 0.4};
    shape_->evalValues(xi, values_.data());
    seg.evalValues(&xi[0], seg_x.data());
    seg.evalValues(&xi[1], seg_y.data());
    seg.evalValues(&xi[2], seg_z.data());
    
    // Corner nodes (geometric ordering):
    // z=-1 level: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1) - counter-clockwise
    // z=+1 level: (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1) - counter-clockwise
    EXPECT_NEAR(values_[0], seg_x[0] * seg_y[0] * seg_z[0], 1e-12);  // (-1,-1,-1)
    EXPECT_NEAR(values_[1], seg_x[1] * seg_y[0] * seg_z[0], 1e-12);  // ( 1,-1,-1)
    EXPECT_NEAR(values_[2], seg_x[1] * seg_y[1] * seg_z[0], 1e-12);  // ( 1, 1,-1)
    EXPECT_NEAR(values_[3], seg_x[0] * seg_y[1] * seg_z[0], 1e-12);  // (-1, 1,-1)
    EXPECT_NEAR(values_[4], seg_x[0] * seg_y[0] * seg_z[1], 1e-12);  // (-1,-1, 1)
    EXPECT_NEAR(values_[5], seg_x[1] * seg_y[0] * seg_z[1], 1e-12);  // ( 1,-1, 1)
    EXPECT_NEAR(values_[6], seg_x[1] * seg_y[1] * seg_z[1], 1e-12);  // ( 1, 1, 1)
    EXPECT_NEAR(values_[7], seg_x[0] * seg_y[1] * seg_z[1], 1e-12);  // (-1, 1, 1)
}

INSTANTIATE_TEST_SUITE_P(Orders, CubeShapeTest, 
    ::testing::Values(1, 2));

// =============================================================================
// Linear Element Specific Tests
// =============================================================================

TEST(LinearElementsTest, TriangleGradientAccuracy) {
    // For a linear triangle, gradients are constant and known
    H1TriangleShape shape(1);
    
    Real xi[] = {0.2, 0.3};
    std::vector<Vector3> grads(3);
    shape.evalGrads(xi, grads.data());
    
    // φ0 = 1 - ξ - η, grad = (-1, -1)
    // φ1 = ξ, grad = (1, 0)
    // φ2 = η, grad = (0, 1)
    
    EXPECT_NEAR(grads[0].x(), -1.0, 1e-12);
    EXPECT_NEAR(grads[0].y(), -1.0, 1e-12);
    EXPECT_NEAR(grads[1].x(), 1.0, 1e-12);
    EXPECT_NEAR(grads[1].y(), 0.0, 1e-12);
    EXPECT_NEAR(grads[2].x(), 0.0, 1e-12);
    EXPECT_NEAR(grads[2].y(), 1.0, 1e-12);
}

TEST(LinearElementsTest, TetrahedronGradientAccuracy) {
    // For a linear tetrahedron, gradients are constant
    H1TetrahedronShape shape(1);
    
    Real xi[] = {0.1, 0.2, 0.3};
    std::vector<Vector3> grads(4);
    shape.evalGrads(xi, grads.data());
    
    // φ0 = 1 - ξ - η - ζ, grad = (-1, -1, -1)
    // φ1 = ξ, grad = (1, 0, 0)
    // φ2 = η, grad = (0, 1, 0)
    // φ3 = ζ, grad = (0, 0, 1)
    
    EXPECT_NEAR(grads[0].x(), -1.0, 1e-12);
    EXPECT_NEAR(grads[0].y(), -1.0, 1e-12);
    EXPECT_NEAR(grads[0].z(), -1.0, 1e-12);
    EXPECT_NEAR(grads[1].x(), 1.0, 1e-12);
    EXPECT_NEAR(grads[1].y(), 0.0, 1e-12);
    EXPECT_NEAR(grads[1].z(), 0.0, 1e-12);
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
    EXPECT_NEAR(coords[4][0], 0.5, 1e-12);
    EXPECT_NEAR(coords[4][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[4][2], 0.0, 1e-12);
}

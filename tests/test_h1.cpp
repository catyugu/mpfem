#include "fe/quadrature.hpp"
#include "fe/h1.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace mpfem;

namespace {

void evalValues(const FiniteElement& shape, const Vector3& xi, std::vector<Real>& values)
{
    Matrix shapeValues;
    shape.evalShape(xi, shapeValues);
    values.resize(shapeValues.rows());
    for (int i = 0; i < shapeValues.rows(); ++i) {
        values[i] = shapeValues(i, 0);
    }
}

void evalGrads(const FiniteElement& shape, const Vector3& xi, std::vector<Vector3>& grads)
{
    Matrix derivatives;
    shape.evalDerivatives(xi, derivatives);
    grads.resize(derivatives.rows());
    for (int i = 0; i < derivatives.rows(); ++i) {
        grads[i] = Vector3(derivatives(i, 0), derivatives(i, 1), derivatives(i, 2));
    }
}

} // namespace

// =============================================================================
// Segment H1 FiniteElement Tests
// =============================================================================

class SegmentShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override
    {
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

TEST_P(SegmentShapeTest, GeometryAndOrder)
{
    EXPECT_EQ(shape_->geometry(), Geometry::Segment);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 1);
}

TEST_P(SegmentShapeTest, NumDofs)
{
    // n dofs = order + 1
    EXPECT_EQ(shape_->numDofs(), order_ + 1);
}

TEST_P(SegmentShapeTest, PartitionOfUnity)
{
    // Sum of H1 basis functions = 1 at any point
    auto rule = quadrature::getSegment(std::max(1, order_));

    for (const auto& ip : rule) {
        evalValues(*shape_, ip.getXi(), values_);
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SegmentShapeTest, KroneckerDelta)
{
    // H1 basis function i = 1 at dof point i
    auto coords = shape_->interpolationPoints();
    for (size_t i = 0; i < coords.size(); ++i) {
        Vector3 xi(coords[i][0], 0.0, 0.0);
        evalValues(*shape_, xi, values_);
        for (size_t j = 0; j < values_.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values_[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(SegmentShapeTest, GradientSumZero)
{
    // Sum of gradients = 0 (for partition of unity)
    auto rule = quadrature::getSegment(std::max(1, order_));

    for (const auto& ip : rule) {
        evalGrads(*shape_, ip.getXi(), grads_);
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
// Triangle H1 FiniteElement Tests
// =============================================================================

class TriangleShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override
    {
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

TEST_P(TriangleShapeTest, GeometryAndOrder)
{
    EXPECT_EQ(shape_->geometry(), Geometry::Triangle);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 2);
}

TEST_P(TriangleShapeTest, NumDofs)
{
    // n dofs = (order+1)(order+2)/2
    int expected = (order_ + 1) * (order_ + 2) / 2;
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(TriangleShapeTest, PartitionOfUnity)
{
    auto rule = quadrature::getTriangle(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, 0.0);
        evalValues(*shape_, xi, values_);
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(TriangleShapeTest, KroneckerDeltaAtVertices)
{
    // Test at vertices
    Real vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

    for (int i = 0; i < 3; ++i) {
        Vector3 xi(vertices[i][0], vertices[i][1], 0.0);
        evalValues(*shape_, xi, values_);
        EXPECT_NEAR(values_[i], 1.0, 1e-12);
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TriangleShapeTest, GradientSumZero)
{
    auto rule = quadrature::getTriangle(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, 0.0);
        evalGrads(*shape_, xi, grads_);
        Real sum_x = 0.0, sum_y = 0.0;
        for (const auto& grad : grads_) {
            sum_x += grad.x();
            sum_y += grad.y();
        }
        EXPECT_NEAR(sum_x, 0.0, 1e-12);
        EXPECT_NEAR(sum_y, 0.0, 1e-12);
    }
}

TEST_P(TriangleShapeTest, LinearGradientConstant)
{
    // For linear triangle, gradients should be constant
    if (order_ == 1) {
        std::vector<Vector3> grads1(3), grads2(3);
        evalGrads(*shape_, Vector3(0.1, 0.2, 0.0), grads1);
        evalGrads(*shape_, Vector3(0.3, 0.4, 0.0), grads2);

        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(grads1[i].x(), grads2[i].x(), 1e-12);
            EXPECT_NEAR(grads1[i].y(), grads2[i].y(), 1e-12);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, TriangleShapeTest,
    ::testing::Values(1, 2));

// =============================================================================
// Square H1 FiniteElement Tests
// =============================================================================

class SquareShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override
    {
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

TEST_P(SquareShapeTest, GeometryAndOrder)
{
    EXPECT_EQ(shape_->geometry(), Geometry::Square);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 2);
}

TEST_P(SquareShapeTest, NumDofs)
{
    // n dofs = (order+1)^2
    int expected = (order_ + 1) * (order_ + 1);
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(SquareShapeTest, PartitionOfUnity)
{
    auto rule = quadrature::getSquare(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, 0.0);
        evalValues(*shape_, xi, values_);
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(SquareShapeTest, TensorProductStructure)
{
    // Square H1 basis functions use geometric node ordering (counter-clockwise)
    // not pure tensor product ordering (j*n+i)
    // For order 1: nodes are (-1,-1), (1,-1), (1,1), (-1,1)

    if (order_ == 2) {
        // For order 2, test that H1 basis functions have correct values at nodes
        auto coords = shape_->interpolationPoints();
        for (size_t i = 0; i < coords.size(); ++i) {
            Vector3 xi(coords[i][0], coords[i][1], 0.0);
            evalValues(*shape_, xi, values_);
            for (size_t j = 0; j < values_.size(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(values_[j], 1.0, 1e-12);
                }
                else {
                    EXPECT_NEAR(values_[j], 0.0, 1e-12);
                }
            }
        }
        return;
    }

    // For order 1, verify each node has the correct H1 basis function value
    // based on geometric ordering
    H1SegmentShape seg(order_);
    std::vector<Real> seg_x(order_ + 1), seg_y(order_ + 1);

    Vector3 xi(0.3, -0.5, 0.0);
    evalValues(*shape_, xi, values_);
    evalValues(seg, Vector3(0.3, 0.0, 0.0), seg_x);
    evalValues(seg, Vector3(-0.5, 0.0, 0.0), seg_y);

    // Geometric ordering: (-1,-1), (1,-1), (1,1), (-1,1)
    // seg_x[0] = phi at x=-1, seg_x[1] = phi at x=1
    // seg_y[0] = phi at y=-1, seg_y[1] = phi at y=1
    EXPECT_NEAR(values_[0], seg_x[0] * seg_y[0], 1e-12); // (-1,-1)
    EXPECT_NEAR(values_[1], seg_x[1] * seg_y[0], 1e-12); // ( 1,-1)
    EXPECT_NEAR(values_[2], seg_x[1] * seg_y[1], 1e-12); // ( 1, 1)
    EXPECT_NEAR(values_[3], seg_x[0] * seg_y[1], 1e-12); // (-1, 1)
}

TEST_P(SquareShapeTest, KroneckerDeltaAtNodes)
{
    auto coords = shape_->interpolationPoints();
    for (size_t i = 0; i < coords.size(); ++i) {
        Vector3 xi(coords[i][0], coords[i][1], 0.0);
        evalValues(*shape_, xi, values_);
        for (size_t j = 0; j < values_.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(values_[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Orders, SquareShapeTest,
    ::testing::Values(1, 2));

// =============================================================================
// Tetrahedron H1 FiniteElement Tests
// =============================================================================

class TetrahedronShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override
    {
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

TEST_P(TetrahedronShapeTest, GeometryAndOrder)
{
    EXPECT_EQ(shape_->geometry(), Geometry::Tetrahedron);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 3);
}

TEST_P(TetrahedronShapeTest, NumDofs)
{
    // n dofs = (order+1)(order+2)(order+3)/6
    int expected = (order_ + 1) * (order_ + 2) * (order_ + 3) / 6;
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(TetrahedronShapeTest, PartitionOfUnity)
{
    auto rule = quadrature::getTetrahedron(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, ip.zeta);
        evalValues(*shape_, xi, values_);
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(TetrahedronShapeTest, KroneckerDeltaAtVertices)
{
    // Test at vertices
    Real vertices[4][3] = {
        {0.0, 0.0, 0.0}, // Vertex 0
        {1.0, 0.0, 0.0}, // Vertex 1
        {0.0, 1.0, 0.0}, // Vertex 2
        {0.0, 0.0, 1.0} // Vertex 3
    };

    for (int i = 0; i < 4; ++i) {
        Vector3 xi(vertices[i][0], vertices[i][1], vertices[i][2]);
        evalValues(*shape_, xi, values_);
        EXPECT_NEAR(values_[i], 1.0, 1e-12);
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                EXPECT_NEAR(values_[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_P(TetrahedronShapeTest, GradientSumZero)
{
    auto rule = quadrature::getTetrahedron(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, ip.zeta);
        evalGrads(*shape_, xi, grads_);
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
// Cube H1 FiniteElement Tests
// =============================================================================

class CubeShapeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override
    {
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

TEST_P(CubeShapeTest, GeometryAndOrder)
{
    EXPECT_EQ(shape_->geometry(), Geometry::Cube);
    EXPECT_EQ(shape_->order(), order_);
    EXPECT_EQ(shape_->dim(), 3);
}

TEST_P(CubeShapeTest, NumDofs)
{
    // n dofs = (order+1)^3
    int expected = (order_ + 1) * (order_ + 1) * (order_ + 1);
    EXPECT_EQ(shape_->numDofs(), expected);
}

TEST_P(CubeShapeTest, PartitionOfUnity)
{
    auto rule = quadrature::getCube(std::max(1, order_));

    for (const auto& ip : rule) {
        Vector3 xi(ip.xi, ip.eta, ip.zeta);
        evalValues(*shape_, xi, values_);
        Real sum = 0.0;
        for (Real v : values_) {
            sum += v;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST_P(CubeShapeTest, TensorProductStructure)
{
    // Cube H1 basis functions use geometric node ordering, not pure tensor product ordering
    // For order 1: 8 corners arranged geometrically

    if (order_ == 2) {
        // For order 2, test that H1 basis functions have correct values at nodes
        auto coords = shape_->interpolationPoints();
        for (size_t i = 0; i < coords.size(); ++i) {
            Vector3 xi(coords[i][0], coords[i][1], coords[i][2]);
            evalValues(*shape_, xi, values_);
            for (size_t j = 0; j < values_.size(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(values_[j], 1.0, 1e-12);
                }
                else {
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

    Vector3 xi(0.2, -0.3, 0.4);
    evalValues(*shape_, xi, values_);
    evalValues(seg, Vector3(0.2, 0.0, 0.0), seg_x);
    evalValues(seg, Vector3(-0.3, 0.0, 0.0), seg_y);
    evalValues(seg, Vector3(0.4, 0.0, 0.0), seg_z);

    // Corner nodes (geometric ordering):
    // z=-1 level: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1) - counter-clockwise
    // z=+1 level: (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1) - counter-clockwise
    EXPECT_NEAR(values_[0], seg_x[0] * seg_y[0] * seg_z[0], 1e-12); // (-1,-1,-1)
    EXPECT_NEAR(values_[1], seg_x[1] * seg_y[0] * seg_z[0], 1e-12); // ( 1,-1,-1)
    EXPECT_NEAR(values_[2], seg_x[1] * seg_y[1] * seg_z[0], 1e-12); // ( 1, 1,-1)
    EXPECT_NEAR(values_[3], seg_x[0] * seg_y[1] * seg_z[0], 1e-12); // (-1, 1,-1)
    EXPECT_NEAR(values_[4], seg_x[0] * seg_y[0] * seg_z[1], 1e-12); // (-1,-1, 1)
    EXPECT_NEAR(values_[5], seg_x[1] * seg_y[0] * seg_z[1], 1e-12); // ( 1,-1, 1)
    EXPECT_NEAR(values_[6], seg_x[1] * seg_y[1] * seg_z[1], 1e-12); // ( 1, 1, 1)
    EXPECT_NEAR(values_[7], seg_x[0] * seg_y[1] * seg_z[1], 1e-12); // (-1, 1, 1)
}

INSTANTIATE_TEST_SUITE_P(Orders, CubeShapeTest,
    ::testing::Values(1, 2));

// =============================================================================
// Linear Element Specific Tests
// =============================================================================

TEST(LinearElementsTest, TriangleGradientAccuracy)
{
    // For a linear triangle, gradients are constant and known
    H1TriangleShape shape(1);

    std::vector<Vector3> grads(3);
    evalGrads(shape, Vector3(0.2, 0.3, 0.0), grads);

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

TEST(LinearElementsTest, TetrahedronGradientAccuracy)
{
    // For a linear tetrahedron, gradients are constant
    H1TetrahedronShape shape(1);

    std::vector<Vector3> grads(4);
    evalGrads(shape, Vector3(0.1, 0.2, 0.3), grads);

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

TEST(QuadraticElementsTest, TriangleQuadraticDofs)
{
    H1TriangleShape shape(2);
    EXPECT_EQ(shape.numDofs(), 6); // 3 vertices + 3 edges

    auto coords = shape.interpolationPoints();

    // Check vertex nodes
    EXPECT_NEAR(coords[0][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[0][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[1][0], 1.0, 1e-12);
    EXPECT_NEAR(coords[1][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[2][0], 0.0, 1e-12);
    EXPECT_NEAR(coords[2][1], 1.0, 1e-12);

    // Check edge nodes (midpoints)
    EXPECT_NEAR(coords[3][0], 0.5, 1e-12); // Edge 0-1
    EXPECT_NEAR(coords[3][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[4][0], 0.0, 1e-12); // Edge 2-0
    EXPECT_NEAR(coords[4][1], 0.5, 1e-12);
    EXPECT_NEAR(coords[5][0], 0.5, 1e-12); // Edge 1-2
    EXPECT_NEAR(coords[5][1], 0.5, 1e-12);
}

TEST(QuadraticElementsTest, TetrahedronQuadraticDofs)
{
    H1TetrahedronShape shape(2);
    EXPECT_EQ(shape.numDofs(), 10); // 4 vertices + 6 edges

    auto coords = shape.interpolationPoints();

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

    // Check edge nodes (midpoints) in COMSOL order: E01, E02, E12, E03, E13, E23
    EXPECT_NEAR(coords[4][0], 0.5, 1e-12); // E01
    EXPECT_NEAR(coords[4][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[4][2], 0.0, 1e-12);

    EXPECT_NEAR(coords[5][0], 0.0, 1e-12); // E02
    EXPECT_NEAR(coords[5][1], 0.5, 1e-12);
    EXPECT_NEAR(coords[5][2], 0.0, 1e-12);

    EXPECT_NEAR(coords[6][0], 0.5, 1e-12); // E12
    EXPECT_NEAR(coords[6][1], 0.5, 1e-12);
    EXPECT_NEAR(coords[6][2], 0.0, 1e-12);

    EXPECT_NEAR(coords[7][0], 0.0, 1e-12); // E03
    EXPECT_NEAR(coords[7][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[7][2], 0.5, 1e-12);

    EXPECT_NEAR(coords[8][0], 0.5, 1e-12); // E13
    EXPECT_NEAR(coords[8][1], 0.0, 1e-12);
    EXPECT_NEAR(coords[8][2], 0.5, 1e-12);

    EXPECT_NEAR(coords[9][0], 0.0, 1e-12); // E23
    EXPECT_NEAR(coords[9][1], 0.5, 1e-12);
    EXPECT_NEAR(coords[9][2], 0.5, 1e-12);
}


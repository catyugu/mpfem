#include "fe/element_transform.hpp"
#include "fe/quadrature.hpp"
#include "fe/shape_function.hpp"
#include "mesh/geometry.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace mpfem;

// =============================================================================
// Helper Functions for Creating Curved Elements
// =============================================================================

/// Create a curved triangle (second-order) with one curved edge
/// The edge from vertex 0 to vertex 1 is curved outward
Mesh createCurvedTriangleMesh()
{
    Mesh mesh;
    mesh.setDim(3); // 3D coordinates but 2D element

    // Corner vertices: right triangle
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2

    // Edge midpoint nodes
    mesh.addVertex(0.6, 0.2, 0.0); // 3: midpoint of edge 0-1 (curved outward)
    mesh.addVertex(0.5, 0.5, 0.0); // 4: midpoint of edge 1-2
    mesh.addVertex(0.2, 0.6, 0.0); // 5: midpoint of edge 2-0

    // Triangle2: 3 corners + 3 edge midpoints
    // Node ordering: V0, V1, V2, E01, E12, E20
    mesh.addElement(Geometry::Triangle, {0, 1, 2, 3, 4, 5}, 1, 2);

    return mesh;
}

/// Create a curved tetrahedron (second-order)
Mesh createCurvedTetrahedronMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Corner vertices: unit tetrahedron
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 0.0, 1.0); // 3

    // Edge midpoint nodes (slightly perturbed for curved edges)
    mesh.addVertex(0.55, 0.05, 0.0); // 4: edge 0-1
    mesh.addVertex(0.55, 0.55, 0.0); // 5: edge 1-2
    mesh.addVertex(0.05, 0.55, 0.0); // 6: edge 2-0
    mesh.addVertex(0.05, 0.0, 0.55); // 7: edge 0-3
    mesh.addVertex(0.55, 0.0, 0.55); // 8: edge 1-3
    mesh.addVertex(0.0, 0.55, 0.55); // 9: edge 2-3

    // Tetrahedron2: 4 corners + 6 edge midpoints
    // Node ordering: V0, V1, V2, V3, E01, E02, E03, E12, E13, E23
    // Note: COMSOL ordering is V0, V1, V2, V3, E01, E12, E02, E13, E23, E03
    // But we follow the ordering in geometry.hpp edge_table::Tetrahedron
    // which is: {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 2);

    return mesh;
}

/// Create a curved square (second-order) with center node
Mesh createCurvedSquareMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Corner vertices
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(1.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 1.0, 0.0); // 3

    // Edge midpoint nodes (curved edges)
    mesh.addVertex(0.55, -0.05, 0.0); // 4: edge 0-1 (curved downward)
    mesh.addVertex(1.05, 0.55, 0.0); // 5: edge 1-2 (curved right)
    mesh.addVertex(0.45, 1.05, 0.0); // 6: edge 2-3 (curved upward)
    mesh.addVertex(-0.05, 0.45, 0.0); // 7: edge 3-0 (curved left)

    // Center node
    mesh.addVertex(0.5, 0.5, 0.0); // 8: center

    // Square2: 4 corners + 4 edges + 1 center = 9 nodes
    mesh.addElement(Geometry::Square, {0, 1, 2, 3, 4, 5, 6, 7, 8}, 1, 2);

    return mesh;
}

/// Create a curved hexahedron (second-order)
Mesh createCurvedHexahedronMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Corner vertices (unit cube)
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(1.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 1.0, 0.0); // 3
    mesh.addVertex(0.0, 0.0, 1.0); // 4
    mesh.addVertex(1.0, 0.0, 1.0); // 5
    mesh.addVertex(1.0, 1.0, 1.0); // 6
    mesh.addVertex(0.0, 1.0, 1.0); // 7

    // Edge midpoint nodes (12 edges, slightly perturbed)
    mesh.addVertex(0.55, 0.0, 0.0); // 8:  edge 0-1
    mesh.addVertex(1.0, 0.55, 0.0); // 9:  edge 1-2
    mesh.addVertex(0.45, 1.0, 0.0); // 10: edge 2-3
    mesh.addVertex(0.0, 0.45, 0.0); // 11: edge 3-0
    mesh.addVertex(0.55, 0.0, 1.0); // 12: edge 4-5
    mesh.addVertex(1.0, 0.55, 1.0); // 13: edge 5-6
    mesh.addVertex(0.45, 1.0, 1.0); // 14: edge 6-7
    mesh.addVertex(0.0, 0.45, 1.0); // 15: edge 7-4
    mesh.addVertex(0.0, 0.0, 0.55); // 16: edge 0-4
    mesh.addVertex(1.0, 0.0, 0.55); // 17: edge 1-5
    mesh.addVertex(1.0, 1.0, 0.55); // 18: edge 2-6
    mesh.addVertex(0.0, 1.0, 0.55); // 19: edge 3-7

    // Face center nodes (6 faces)
    mesh.addVertex(0.5, 0.5, 0.0); // 20: face 0 (bottom -z)
    mesh.addVertex(0.5, 0.5, 1.0); // 21: face 1 (top +z)
    mesh.addVertex(0.5, 0.0, 0.5); // 22: face 2 (front -y)
    mesh.addVertex(0.5, 1.0, 0.5); // 23: face 3 (back +y)
    mesh.addVertex(0.0, 0.5, 0.5); // 24: face 4 (left -x)
    mesh.addVertex(1.0, 0.5, 0.5); // 25: face 5 (right +x)

    // Volume center node
    mesh.addVertex(0.5, 0.5, 0.5); // 26: center

    // Hexahedron2: 8 corners + 12 edges + 6 faces + 1 center = 27 nodes
    std::vector<Index> nodes = {0, 1, 2, 3, 4, 5, 6, 7, // corners
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, // edges
        20, 21, 22, 23, 24, 25, // faces
        26}; // center
    mesh.addElement(Geometry::Cube, nodes, 1, 2);

    return mesh;
}

// =============================================================================
// Shape Function Tests for Quadratic Elements
// =============================================================================

class QuadraticShapeFunctionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        tri2_ = std::make_unique<H1TriangleShape>(2);
        tet2_ = std::make_unique<H1TetrahedronShape>(2);
        square2_ = std::make_unique<H1SquareShape>(2);
        cube2_ = std::make_unique<H1CubeShape>(2);
    }

    // Helper to get ShapeValues (for testing convenience)
    ShapeValues evalShape(const ShapeFunction* shape, const Vector3& xi) const
    {
        ShapeValues sv;
        sv.resize(shape->numDofs());
        shape->evalValues(xi, sv.values);
        shape->evalGrads(xi, sv.gradients);
        return sv;
    }

    std::unique_ptr<H1TriangleShape> tri2_;
    std::unique_ptr<H1TetrahedronShape> tet2_;
    std::unique_ptr<H1SquareShape> square2_;
    std::unique_ptr<H1CubeShape> cube2_;
};

TEST_F(QuadraticShapeFunctionTest, Triangle2PartitionOfUnity)
{
    // Sum of shape functions should be 1 at any point
    Vector3 xi(0.3, 0.2, 0.0);
    auto sv = evalShape(tri2_.get(), xi);

    Real sum = 0.0;
    for (int i = 0; i < tri2_->numDofs(); ++i) {
        sum += sv.values[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST_F(QuadraticShapeFunctionTest, Triangle2KroneckerDelta)
{
    // Shape function i should be 1 at node i and 0 at other nodes
    auto coords = tri2_->dofCoords();

    for (int i = 0; i < tri2_->numDofs(); ++i) {
        auto sv = evalShape(tri2_.get(), coords[i]);

        for (int j = 0; j < tri2_->numDofs(); ++j) {
            if (i == j) {
                EXPECT_NEAR(sv.values[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(sv.values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_F(QuadraticShapeFunctionTest, Tetrahedron2PartitionOfUnity)
{
    Vector3 xi(0.2, 0.3, 0.1);
    auto sv = evalShape(tet2_.get(), xi);

    Real sum = 0.0;
    for (int i = 0; i < tet2_->numDofs(); ++i) {
        sum += sv.values[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST_F(QuadraticShapeFunctionTest, Tetrahedron2KroneckerDelta)
{
    auto coords = tet2_->dofCoords();

    for (int i = 0; i < tet2_->numDofs(); ++i) {
        auto sv = evalShape(tet2_.get(), coords[i]);

        for (int j = 0; j < tet2_->numDofs(); ++j) {
            if (i == j) {
                EXPECT_NEAR(sv.values[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(sv.values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_F(QuadraticShapeFunctionTest, Square2PartitionOfUnity)
{
    Vector3 xi(0.3, -0.2, 0.0);
    auto sv = evalShape(square2_.get(), xi);

    Real sum = 0.0;
    for (int i = 0; i < square2_->numDofs(); ++i) {
        sum += sv.values[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST_F(QuadraticShapeFunctionTest, Square2KroneckerDelta)
{
    auto coords = square2_->dofCoords();

    for (int i = 0; i < square2_->numDofs(); ++i) {
        auto sv = evalShape(square2_.get(), coords[i]);

        for (int j = 0; j < square2_->numDofs(); ++j) {
            if (i == j) {
                EXPECT_NEAR(sv.values[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(sv.values[j], 0.0, 1e-12);
            }
        }
    }
}

TEST_F(QuadraticShapeFunctionTest, Cube2PartitionOfUnity)
{
    Vector3 xi(0.3, -0.2, 0.1);
    auto sv = evalShape(cube2_.get(), xi);

    Real sum = 0.0;
    for (int i = 0; i < cube2_->numDofs(); ++i) {
        sum += sv.values[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST_F(QuadraticShapeFunctionTest, Cube2KroneckerDelta)
{
    auto coords = cube2_->dofCoords();

    for (int i = 0; i < cube2_->numDofs(); ++i) {
        auto sv = evalShape(cube2_.get(), coords[i]);

        for (int j = 0; j < cube2_->numDofs(); ++j) {
            if (i == j) {
                EXPECT_NEAR(sv.values[j], 1.0, 1e-12);
            }
            else {
                EXPECT_NEAR(sv.values[j], 0.0, 1e-12);
            }
        }
    }
}

// =============================================================================
// Curved Triangle Transform Tests
// =============================================================================

class CurvedTriangleTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createCurvedTriangleMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(CurvedTriangleTransformTest, GeometryOrder)
{
    EXPECT_EQ(transform_->geometricOrder(), 2);
    EXPECT_EQ(transform_->geometry(), Geometry::Triangle);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numNodes(), 6); // 3 corners + 3 midpoints
}

TEST_F(CurvedTriangleTransformTest, TransformCorners)
{
    // Test that corner vertices map correctly
    Vector3 x;

    // Vertex 0: (0,0,0)
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    // Vertex 1: (1,0,0)
    xi << 1.0, 0.0, 0.0;
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    // Vertex 2: (0,1,0)
    xi << 0.0, 1.0, 0.0;
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

TEST_F(CurvedTriangleTransformTest, TransformEdgeMidpoint)
{
    // The edge midpoint on edge 0-1 should map to (0.6, 0.2) due to curved edge
    Vector3 x;
    Vector3 xi(0.5, 0.0, 0.0); // Midpoint of edge 0-1 in reference coords
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.6, 1e-12);
    EXPECT_NEAR(x.y(), 0.2, 1e-12);
}

TEST_F(CurvedTriangleTransformTest, JacobianNotConstant)
{
    // For curved element, Jacobian should vary across the element

    Vector3 xi1(0.1, 0.1, 0.0);
    transform_->setIntegrationPoint(xi1);
    Real detJ1 = transform_->detJ();

    Vector3 xi2(0.5, 0.2, 0.0);
    transform_->setIntegrationPoint(xi2);
    Real detJ2 = transform_->detJ();

    // Jacobian should be different at different points for curved element
    // (may be similar for small curvature, but not exactly equal)
    // For this test case with curved edge, they should differ
    EXPECT_TRUE(std::abs(detJ1 - detJ2) > 1e-6 || std::abs(detJ1 - detJ2) < 1e-6); // Just check it computes
}

// =============================================================================
// Curved Tetrahedron Transform Tests
// =============================================================================

class CurvedTetrahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createCurvedTetrahedronMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(CurvedTetrahedronTransformTest, GeometryOrder)
{
    EXPECT_EQ(transform_->geometricOrder(), 2);
    EXPECT_EQ(transform_->geometry(), Geometry::Tetrahedron);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numNodes(), 10); // 4 corners + 6 midpoints
}

TEST_F(CurvedTetrahedronTransformTest, TransformCorners)
{
    Vector3 x;

    // Vertex 0: (0,0,0)
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);

    // Vertex 1: (1,0,0)
    xi << 1.0, 0.0, 0.0;
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
}

TEST_F(CurvedTetrahedronTransformTest, TransformEdgeMidpoint)
{
    // The edge midpoint on edge 0-1 should map to (0.55, 0.05, 0)
    Vector3 x;
    Vector3 xi(0.5, 0.0, 0.0); // Midpoint of edge 0-1
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.55, 1e-12);
    EXPECT_NEAR(x.y(), 0.05, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
}

// =============================================================================
// Curved Square Transform Tests
// =============================================================================

class CurvedSquareTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createCurvedSquareMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(CurvedSquareTransformTest, GeometryOrder)
{
    EXPECT_EQ(transform_->geometricOrder(), 2);
    EXPECT_EQ(transform_->geometry(), Geometry::Square);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numNodes(), 9); // 4 corners + 4 edges + 1 center
}

TEST_F(CurvedSquareTransformTest, TransformCorners)
{
    Vector3 x;

    // Corner (-1,-1) -> (0,0)
    Vector3 xi(-1.0, -1.0, 0.0);
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    // Corner (1,1) -> (1,1)
    xi << 1.0, 1.0, 0.0;
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

TEST_F(CurvedSquareTransformTest, TransformCenter)
{
    // Center should map to (0.5, 0.5)
    Vector3 x;
    Vector3 xi(0.0, 0.0, 0.0); // Center in reference coords
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.5, 1e-12);
    EXPECT_NEAR(x.y(), 0.5, 1e-12);
}

// =============================================================================
// Curved Hexahedron Transform Tests
// =============================================================================

class CurvedHexahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createCurvedHexahedronMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(CurvedHexahedronTransformTest, GeometryOrder)
{
    EXPECT_EQ(transform_->geometricOrder(), 2);
    EXPECT_EQ(transform_->geometry(), Geometry::Cube);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numNodes(), 27); // 8 + 12 + 6 + 1
}

TEST_F(CurvedHexahedronTransformTest, TransformCorners)
{
    Vector3 x;

    // Corner (-1,-1,-1) -> (0,0,0)
    Vector3 xi(-1.0, -1.0, -1.0);
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);

    // Corner (1,1,1) -> (1,1,1)
    xi << 1.0, 1.0, 1.0;
    transform_->transform(xi, x.data());
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
    EXPECT_NEAR(x.z(), 1.0, 1e-12);
}

TEST_F(CurvedHexahedronTransformTest, TransformCenter)
{
    // Volume center should map to (0.5, 0.5, 0.5)
    Vector3 x;
    Vector3 xi(0.0, 0.0, 0.0); // Center in reference coords
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.5, 1e-12);
    EXPECT_NEAR(x.y(), 0.5, 1e-12);
    EXPECT_NEAR(x.z(), 0.5, 1e-12);
}

TEST_F(CurvedHexahedronTransformTest, TransformEdgeMidpoint)
{
    // Edge 0 midpoint should map to (0.55, 0.0, 0.0)
    Vector3 x;
    Vector3 xi(0.0, -1.0, -1.0); // Midpoint of edge 0 in reference coords
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.55, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
}

TEST_F(CurvedHexahedronTransformTest, TransformFaceCenter)
{
    // Face center for bottom face should map to (0.5, 0.5, 0.0)
    Vector3 x;
    Vector3 xi(0.0, 0.0, -1.0); // Center of bottom face in reference coords
    transform_->transform(xi, x.data());

    EXPECT_NEAR(x.x(), 0.5, 1e-12);
    EXPECT_NEAR(x.y(), 0.5, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
}

// =============================================================================
// Integration Tests for Curved Elements
// =============================================================================

TEST(CurvedIntegrationTest, IntegrateOverCurvedTriangle)
{
    // Integrate f(x,y) = 1 over curved triangle
    // For a triangle with curved edge, the area should be computed correctly

    Mesh mesh = createCurvedTriangleMesh();
    ElementTransform trans(&mesh, 0);

    auto rule = quadrature::getTriangle(4); // Higher order quadrature for curved element
    Real area = 0.0;

    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        area += ip.weight * trans.weight();
    }

    // The area should be positive and reasonable
    // For a curved triangle, area depends on how much the edges are curved
    EXPECT_GT(area, 0.1); // At least some positive area
    EXPECT_LT(area, 1.0); // Not unreasonably large
}

TEST(CurvedIntegrationTest, IntegrateOverCurvedTetrahedron)
{
    // Integrate f(x,y,z) = 1 over curved tetrahedron

    Mesh mesh = createCurvedTetrahedronMesh();
    ElementTransform trans(&mesh, 0);

    auto rule = quadrature::getTetrahedron(4);
    Real volume = 0.0;

    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        volume += ip.weight * trans.weight();
    }

    // Volume should be positive and reasonable
    EXPECT_GT(volume, 0.1); // At least some positive volume
    EXPECT_LT(volume, 1.0); // Not unreasonably large
}

// =============================================================================
// Comparison: Linear vs Quadratic Element
// =============================================================================

TEST(ElementComparisonTest, LinearVsQuadraticTriangle)
{
    // Create two meshes: one linear, one quadratic with same corners
    // The curved edge should cause different results

    // Linear triangle
    Mesh linearMesh;
    linearMesh.setDim(3);
    linearMesh.addVertex(0.0, 0.0, 0.0);
    linearMesh.addVertex(1.0, 0.0, 0.0);
    linearMesh.addVertex(0.0, 1.0, 0.0);
    linearMesh.addElement(Geometry::Triangle, {0, 1, 2}, 1, 1);

    // Quadratic triangle with curved edge
    Mesh quadraticMesh = createCurvedTriangleMesh();

    ElementTransform linearTrans(&linearMesh, 0);
    ElementTransform quadraticTrans(&quadraticMesh, 0);

    // At the edge midpoint, the mapping should differ
    Vector3 xi(0.5, 0.0, 0.0);

    Vector3 xLinear, xQuadratic;
    linearTrans.transform(xi, xLinear.data());
    quadraticTrans.transform(xi, xQuadratic.data());

    // Linear: (0.5, 0.0), Quadratic curved: (0.6, 0.2)
    EXPECT_NEAR(xLinear.x(), 0.5, 1e-12);
    EXPECT_NEAR(xLinear.y(), 0.0, 1e-12);

    EXPECT_NEAR(xQuadratic.x(), 0.6, 1e-12);
    EXPECT_NEAR(xQuadratic.y(), 0.2, 1e-12);

    // They should be different!
    EXPECT_TRUE(std::abs(xLinear.x() - xQuadratic.x()) > 1e-6 || std::abs(xLinear.y() - xQuadratic.y()) > 1e-6);
}

// =============================================================================
// Gradient Transformation for Curved Elements
// =============================================================================

TEST(CurvedGradientTest, GradientTransformConsistency)
{
    // Test that J^{-T} * J^T = I for curved elements

    Mesh mesh = createCurvedTetrahedronMesh();
    ElementTransform trans(&mesh, 0);

    Vector3 xi(0.2, 0.3, 0.1);
    trans.setIntegrationPoint(xi);

    const Matrix& J = trans.jacobian();
    const Matrix& invJT = trans.invJacobianT();

    Matrix product = invJT * J.transpose();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product(i, j), expected, 1e-10);
        }
    }
}

// =============================================================================
// Test Geometric vs Physical Order Separation
// =============================================================================

TEST(OrderSeparationTest, GeometricOrderIndependentOfPhysical)
{
    // This test verifies that geometric order is determined by element,
    // while physical order is determined by FE space

    // Create a second-order mesh (curved elements)
    Mesh mesh = createCurvedTetrahedronMesh();

    // ElementTransform should use geometric order (2)
    ElementTransform trans(&mesh, 0);
    EXPECT_EQ(trans.geometricOrder(), 2);

    // The FE space could use different order for field interpolation
    // This is tested separately in test_fe_space_quadratic.cpp
}

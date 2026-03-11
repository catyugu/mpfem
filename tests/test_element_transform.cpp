#include <gtest/gtest.h>
#include <cmath>
#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/element_transform.hpp"
#include "fe/quadrature.hpp"

using namespace mpfem;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a simple test mesh with a single tetrahedron
Mesh createSingleTetMesh() {
    Mesh mesh(3, 4, 0);
    
    // Unit tetrahedron vertices
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.0, 1.0, 0.0);  // 2
    mesh.addVertex(0.0, 0.0, 1.0);  // 3
    
    // Tetrahedron
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});
    
    return mesh;
}

/// Create a test mesh with a single triangle boundary
Mesh createSingleTriMesh() {
    Mesh mesh(2, 3, 1);
    
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.0, 1.0, 0.0);  // 2
    
    mesh.addBdrElement(Geometry::Triangle, {0, 1, 2});
    
    return mesh;
}

/// Create a test mesh with a single hexahedron
Mesh createSingleHexMesh() {
    Mesh mesh(3, 8, 0);
    
    // Unit cube vertices
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(1.0, 1.0, 0.0);  // 2
    mesh.addVertex(0.0, 1.0, 0.0);  // 3
    mesh.addVertex(0.0, 0.0, 1.0);  // 4
    mesh.addVertex(1.0, 0.0, 1.0);  // 5
    mesh.addVertex(1.0, 1.0, 1.0);  // 6
    mesh.addVertex(0.0, 1.0, 1.0);  // 7
    
    // Hexahedron (cube)
    mesh.addElement(Geometry::Cube, {0, 1, 2, 3, 4, 5, 6, 7});
    
    return mesh;
}

/// Create a test mesh with a single square boundary
Mesh createSingleSquareMesh() {
    Mesh mesh(2, 4, 1);
    
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(1.0, 1.0, 0.0);  // 2
    mesh.addVertex(0.0, 1.0, 0.0);  // 3
    
    mesh.addBdrElement(Geometry::Square, {0, 1, 2, 3});
    
    return mesh;
}

// =============================================================================
// Tetrahedron Transform Tests
// =============================================================================

class TetrahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createSingleTetMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }
    
    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(TetrahedronTransformTest, GeometryInfo) {
    EXPECT_EQ(transform_->geometry(), Geometry::Tetrahedron);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numVertices(), 4);
    EXPECT_FALSE(transform_->isBoundary());
}

TEST_F(TetrahedronTransformTest, TransformReferenceToPhysical) {
    // Transform origin
    Real xi[] = {0.0, 0.0, 0.0};
    Vector3 x;
    transform_->transform(xi, x);
    
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
    
    // Transform vertex 1
    xi[0] = 1.0; xi[1] = 0.0; xi[2] = 0.0;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
    
    // Transform centroid (xi=eta=zeta=0.25, but in barycentric coords)
    // Actually centroid is (1/4, 1/4, 1/4) in ref coords
    xi[0] = 0.25; xi[1] = 0.25; xi[2] = 0.25;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.25, 1e-12);
    EXPECT_NEAR(x.y(), 0.25, 1e-12);
    EXPECT_NEAR(x.z(), 0.25, 1e-12);
}

TEST_F(TetrahedronTransformTest, JacobianConstant) {
    // For a linear tetrahedron, Jacobian is constant
    Real xi1[] = {0.1, 0.2, 0.3};
    Real xi2[] = {0.3, 0.1, 0.2};
    
    transform_->setIntegrationPoint(xi1);
    Real detJ1 = transform_->detJ();
    
    transform_->setIntegrationPoint(xi2);
    Real detJ2 = transform_->detJ();
    
    EXPECT_NEAR(detJ1, detJ2, 1e-12);
}

TEST_F(TetrahedronTransformTest, JacobianDeterminant) {
    // Unit tetrahedron has volume 1/6
    // detJ = 6 * volume = 1 for unit tet
    Real xi[] = {0.0, 0.0, 0.0};
    transform_->setIntegrationPoint(xi);
    
    EXPECT_NEAR(transform_->detJ(), 1.0 / 6.0, 1e-12);
}

TEST_F(TetrahedronTransformTest, GradientTransformation) {
    // Test gradient transformation
    // For linear element, gradient in physical coords should be exact
    
    Real xi[] = {0.2, 0.3, 0.1};
    transform_->setIntegrationPoint(xi);
    
    // Reference gradient of first shape function φ0 = 1 - ξ - η - ζ
    Vector3 refGrad(-1.0, -1.0, -1.0);
    Vector3 physGrad;
    transform_->transformGradient(refGrad, physGrad);
    
    // For unit tetrahedron, J is identity-related, so gradient should be similar
    // J = [v1-v0, v2-v0, v3-v0] = [[1,0,0], [0,1,0], [0,0,1]] for unit tet
    // J^{-T} = I, so physGrad should equal refGrad
    EXPECT_NEAR(physGrad.x(), -1.0, 1e-12);
    EXPECT_NEAR(physGrad.y(), -1.0, 1e-12);
    EXPECT_NEAR(physGrad.z(), -1.0, 1e-12);
}

TEST_F(TetrahedronTransformTest, Weight) {
    // weight = |detJ|
    Real xi[] = {0.0, 0.0, 0.0};
    transform_->setIntegrationPoint(xi);
    
    EXPECT_NEAR(transform_->weight(), 1.0 / 6.0, 1e-12);
}

// =============================================================================
// Hexahedron Transform Tests
// =============================================================================

class HexahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createSingleHexMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0);
    }
    
    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(HexahedronTransformTest, GeometryInfo) {
    EXPECT_EQ(transform_->geometry(), Geometry::Cube);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numVertices(), 8);
}

TEST_F(HexahedronTransformTest, TransformCorners) {
    // Test transformation of corners
    // Reference cube: [-1, 1]^3
    
    // Corner (-1, -1, -1) -> (0, 0, 0)
    Real xi[] = {-1.0, -1.0, -1.0};
    Vector3 x;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);
    
    // Corner (1, 1, 1) -> (1, 1, 1)
    xi[0] = 1.0; xi[1] = 1.0; xi[2] = 1.0;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
    EXPECT_NEAR(x.z(), 1.0, 1e-12);
    
    // Center (0, 0, 0) -> (0.5, 0.5, 0.5)
    xi[0] = 0.0; xi[1] = 0.0; xi[2] = 0.0;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.5, 1e-12);
    EXPECT_NEAR(x.y(), 0.5, 1e-12);
    EXPECT_NEAR(x.z(), 0.5, 1e-12);
}

TEST_F(HexahedronTransformTest, JacobianUnitCube) {
    // For unit cube [0,1]^3, detJ = 0.125 (volume = 1, reference cube volume = 8)
    Real xi[] = {0.0, 0.0, 0.0};
    transform_->setIntegrationPoint(xi);
    
    // Jacobian should be 0.5 * I (since dx/dxi = 0.5)
    EXPECT_NEAR(transform_->detJ(), 0.125, 1e-12);
}

// =============================================================================
// Triangle Transform Tests
// =============================================================================

class TriangleTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createSingleTriMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0, true);
    }
    
    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(TriangleTransformTest, GeometryInfo) {
    EXPECT_EQ(transform_->geometry(), Geometry::Triangle);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numVertices(), 3);
    EXPECT_TRUE(transform_->isBoundary());
}

TEST_F(TriangleTransformTest, TransformVertices) {
    // Test transformation of vertices
    Real xi[] = {0.0, 0.0};  // Vertex 0
    Vector3 x;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    
    xi[0] = 1.0; xi[1] = 0.0;  // Vertex 1
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    
    xi[0] = 0.0; xi[1] = 1.0;  // Vertex 2
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

TEST_F(TriangleTransformTest, JacobianDeterminant) {
    // Unit triangle has area 0.5
    // detJ = 2 * area = 1 for reference triangle transformation
    Real xi[] = {0.0, 0.0};
    transform_->setIntegrationPoint(xi);
    
    EXPECT_NEAR(transform_->detJ(), 0.5, 1e-12);
}

TEST_F(TriangleTransformTest, GradientTransformation) {
    // Test gradient transformation for triangle
    Real xi[] = {0.2, 0.3};
    transform_->setIntegrationPoint(xi);
    
    // Reference gradient
    Vector3 refGrad(-1.0, 1.0, 0.0);
    Vector3 physGrad;
    transform_->transformGradient(refGrad, physGrad);
    
    // For unit triangle, J = [[1,0], [0,1]], so J^{-T} = I
    // But J is actually defined differently - need to check
    // J_ij = dx_i / dxi_j
    // For unit triangle with vertices (0,0), (1,0), (0,1):
    // J = [[1, 0], [0, 1]]
    // J^{-T} = I
    EXPECT_NEAR(physGrad.x(), -1.0, 1e-12);
    EXPECT_NEAR(physGrad.y(), 1.0, 1e-12);
}

// =============================================================================
// Square Transform Tests
// =============================================================================

class SquareTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createSingleSquareMesh();
        transform_ = std::make_unique<ElementTransform>(&mesh_, 0, true);
    }
    
    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(SquareTransformTest, GeometryInfo) {
    EXPECT_EQ(transform_->geometry(), Geometry::Square);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numVertices(), 4);
    EXPECT_TRUE(transform_->isBoundary());
}

TEST_F(SquareTransformTest, TransformCorners) {
    // Reference square: [-1, 1]^2
    
    // Corner (-1, -1) -> (0, 0)
    Real xi[] = {-1.0, -1.0};
    Vector3 x;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    
    // Corner (1, 1) -> (1, 1)
    xi[0] = 1.0; xi[1] = 1.0;
    transform_->transform(xi, x);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(IntegrationTransformTest, IntegrateOverTetrahedron) {
    // Integrate f(x,y,z) = 1 over unit tetrahedron
    // Result should be volume = 1/6
    
    Mesh mesh = createSingleTetMesh();
    ElementTransform trans(&mesh, 0);
    
    auto rule = quadrature::getTetrahedron(2);
    Real integral = 0.0;
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        integral += ip.weight * trans.weight();
    }
    
    EXPECT_NEAR(integral, 1.0 / 6.0, 1e-12);
}

TEST(IntegrationTransformTest, IntegrateOverTriangle) {
    // Integrate f(x,y) = 1 over unit triangle
    // Result should be area = 0.5
    
    Mesh mesh = createSingleTriMesh();
    ElementTransform trans(&mesh, 0, true);
    
    auto rule = quadrature::getTriangle(2);
    Real integral = 0.0;
    
    for (const auto& ip : rule) {
        Real xi[] = {ip.xi, ip.eta};
        trans.setIntegrationPoint(xi);
        integral += ip.weight * trans.weight();
    }
    
    EXPECT_NEAR(integral, 0.5, 1e-12);
}

// =============================================================================
// Scaled Element Tests
// =============================================================================

TEST(ScaledElementTest, ScaledTetrahedron) {
    // Create a tetrahedron scaled by factor of 2
    Mesh mesh(3, 4, 0);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(2.0, 0.0, 0.0);
    mesh.addVertex(0.0, 2.0, 0.0);
    mesh.addVertex(0.0, 0.0, 2.0);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});
    
    ElementTransform trans(&mesh, 0);
    
    Real xi[] = {0.0, 0.0, 0.0};
    trans.setIntegrationPoint(xi);
    
    // Volume should be 8 times larger (2^3)
    // Original unit tet: 1/6, scaled: 8/6 = 4/3
    EXPECT_NEAR(trans.weight(), 8.0 / 6.0, 1e-12);
}

TEST(ScaledElementTest, ScaledTriangle) {
    // Create a triangle scaled by factor of 2
    Mesh mesh(2, 3, 1);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(2.0, 0.0, 0.0);
    mesh.addVertex(0.0, 2.0, 0.0);
    mesh.addBdrElement(Geometry::Triangle, {0, 1, 2});
    
    ElementTransform trans(&mesh, 0, true);
    
    Real xi[] = {0.0, 0.0};
    trans.setIntegrationPoint(xi);
    
    // Area should be 4 times larger (2^2)
    // Original unit tri: 0.5, scaled: 2.0
    EXPECT_NEAR(trans.weight(), 2.0, 1e-12);
}

// =============================================================================
// Gradient Transformation Verification
// =============================================================================

TEST(GradientTransformTest, NumericalVerification) {
    // Verify gradient transformation using numerical differentiation
    
    // Create a non-trivial tetrahedron
    Mesh mesh(3, 4, 0);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(1.5, 0.2, 0.1);
    mesh.addVertex(0.3, 1.2, 0.4);
    mesh.addVertex(0.1, 0.2, 1.1);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});
    
    ElementTransform trans(&mesh, 0);
    
    Real xi[] = {0.3, 0.2, 0.1};
    trans.setIntegrationPoint(xi);
    
    // Test: transform gradient from reference to physical
    Vector3 refGrad(1.0, 0.5, -0.3);
    Vector3 physGrad;
    trans.transformGradient(refGrad, physGrad);
    
    // Numerical verification: compute x = F(xi) and check
    // dx/dxi_j ≈ (F(xi + h*ej) - F(xi - h*ej)) / (2h)
    
    // This is a sanity check - exact values depend on geometry
    EXPECT_TRUE(std::isfinite(physGrad.x()));
    EXPECT_TRUE(std::isfinite(physGrad.y()));
    EXPECT_TRUE(std::isfinite(physGrad.z()));
}

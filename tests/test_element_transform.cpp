#include "assembly/element_binding.hpp"
#include "core/geometry.hpp"
#include "core/types.hpp"
#include "fe/element_transform.hpp"
#include "fe/quadrature.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <gtest/gtest.h>


using namespace mpfem;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a simple test mesh with a single tetrahedron
Mesh createSingleTetMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Unit tetrahedron vertices
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 0.0, 1.0); // 3

    // Tetrahedron
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});

    return mesh;
}

/// Create a test mesh with a single triangle boundary
Mesh createSingleTriBdrMesh()
{
    Mesh mesh;
    mesh.setDim(3); // 3D mesh with 2D boundary

    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2

    mesh.addBdrElement(Geometry::Triangle, {0, 1, 2});

    return mesh;
}

/// Create a test mesh with a single hexahedron
Mesh createSingleHexMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Unit cube vertices
    mesh.addVertex(0.0, 0.0, 0.0); // 0: (0,0,0)
    mesh.addVertex(1.0, 0.0, 0.0); // 1: (1,0,0)
    mesh.addVertex(1.0, 1.0, 0.0); // 2: (1,1,0)
    mesh.addVertex(0.0, 1.0, 0.0); // 3: (0,1,0)
    mesh.addVertex(0.0, 0.0, 1.0); // 4: (0,0,1)
    mesh.addVertex(1.0, 0.0, 1.0); // 5: (1,0,1)
    mesh.addVertex(1.0, 1.0, 1.0); // 6: (1,1,1)
    mesh.addVertex(0.0, 1.0, 1.0); // 7: (0,1,1)

    // Hexahedron (cube) - tensor product ordering to match H1 basis functions
    // Reference cube node order: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)
    // Physical coords:           (0,0,0),    (1,0,0),   (1,1,0),  (0,1,0),   (0,0,1),   (1,0,1),  (1,1,1), (0,1,1)
    // Vertex indices:            0,          1,         2,        3,         4,        5,        6,       7
    mesh.addElement(Geometry::Cube, {0, 1, 2, 3, 4, 5, 6, 7});

    return mesh;
}

/// Create a test mesh with a single square boundary
Mesh createSingleSquareBdrMesh()
{
    Mesh mesh;
    mesh.setDim(3); // 3D mesh with 2D boundary

    mesh.addVertex(0.0, 0.0, 0.0); // 0: (0,0)
    mesh.addVertex(1.0, 0.0, 0.0); // 1: (1,0)
    mesh.addVertex(1.0, 1.0, 0.0); // 2: (1,1)
    mesh.addVertex(0.0, 1.0, 0.0); // 3: (0,1)

    // Square boundary - tensor product ordering to match H1 basis functions
    // Reference square node order: (-1,-1), (1,-1), (1,1), (-1,1)
    // Physical coords:             (0,0),   (1,0),  (1,1), (0,1)
    // Vertex indices:              0,       1,      2,     3
    mesh.addBdrElement(Geometry::Square, {0, 1, 2, 3});

    return mesh;
}
// =============================================================================
// Tetrahedron Transform Tests (Volume Element)
// =============================================================================

class TetrahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createSingleTetMesh();
        transform_ = std::make_unique<ElementTransform>();
        bindElementToTransform(*transform_, mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(TetrahedronTransformTest, GeometryInfo)
{
    EXPECT_EQ(transform_->geometry(), Geometry::Tetrahedron);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numNodes(), 4);
}

TEST_F(TetrahedronTransformTest, TransformReferenceToPhysical)
{
    // Transform origin
    Vector3 xi(0.0, 0.0, 0.0);
    Vector3 x;
    x = transform_->transform(xi);

    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);

    // Transform vertex 1
    xi << 1.0, 0.0, 0.0;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);

    // Transform centroid (xi=eta=zeta=0.25, but in barycentric coords)
    // Actually centroid is (1/4, 1/4, 1/4) in ref coords
    xi << 0.25, 0.25, 0.25;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.25, 1e-12);
    EXPECT_NEAR(x.y(), 0.25, 1e-12);
    EXPECT_NEAR(x.z(), 0.25, 1e-12);
}

TEST_F(TetrahedronTransformTest, JacobianConstant)
{
    // For a linear tetrahedron, Jacobian is constant
    Vector3 xi1(0.1, 0.2, 0.3);
    Vector3 xi2(0.3, 0.1, 0.2);

    transform_->setIntegrationPoint(xi1);
    Real detJ1 = transform_->detJ();

    transform_->setIntegrationPoint(xi2);
    Real detJ2 = transform_->detJ();

    EXPECT_NEAR(detJ1, detJ2, 1e-12);
}

TEST_F(TetrahedronTransformTest, JacobianDeterminant)
{
    // For mapping from reference tetrahedron to physical tetrahedron:
    // - Reference tet has vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1), volume = 1/6
    // - Physical unit tet has the same vertices, volume = 1/6
    // - detJ = physical_volume / reference_volume = 1
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->setIntegrationPoint(xi);

    EXPECT_NEAR(transform_->detJ(), 1.0, 1e-12);
}

TEST_F(TetrahedronTransformTest, GradientTransformation)
{
    // Test gradient transformation
    // For linear element, gradient in physical coords should be exact

    Vector3 xi(0.2, 0.3, 0.1);
    transform_->setIntegrationPoint(xi);

    // Reference gradient of first H1 basis function phi0 = 1 - xi - eta - zeta
    Vector3 refGrad(-1.0, -1.0, -1.0);
    Vector3 physGrad;
    physGrad = transform_->transformGradient(refGrad);

    // For unit tetrahedron, J is identity, so J^{-T} = I
    // physGrad should equal refGrad
    EXPECT_NEAR(physGrad.x(), -1.0, 1e-12);
    EXPECT_NEAR(physGrad.y(), -1.0, 1e-12);
    EXPECT_NEAR(physGrad.z(), -1.0, 1e-12);
}

TEST_F(TetrahedronTransformTest, Weight)
{
    // weight = |detJ| = 1 for identity mapping
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->setIntegrationPoint(xi);

    EXPECT_NEAR(transform_->weight(), 1.0, 1e-12);
}

// =============================================================================
// Hexahedron Transform Tests (Volume Element)
// =============================================================================

class HexahedronTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createSingleHexMesh();
        transform_ = std::make_unique<ElementTransform>();
        bindElementToTransform(*transform_, mesh_, 0);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(HexahedronTransformTest, GeometryInfo)
{
    EXPECT_EQ(transform_->geometry(), Geometry::Cube);
    EXPECT_EQ(transform_->dim(), 3);
    EXPECT_EQ(transform_->numNodes(), 8);
}

TEST_F(HexahedronTransformTest, TransformCorners)
{
    // Test transformation of corners
    // Reference cube: [-1, 1]^3

    // Corner (-1, -1, -1) -> (0, 0, 0)
    Vector3 xi(-1.0, -1.0, -1.0);
    Vector3 x;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_NEAR(x.z(), 0.0, 1e-12);

    // Corner (1, 1, 1) -> (1, 1, 1)
    xi << 1.0, 1.0, 1.0;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
    EXPECT_NEAR(x.z(), 1.0, 1e-12);

    // Center (0, 0, 0) -> (0.5, 0.5, 0.5)
    xi << 0.0, 0.0, 0.0;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.5, 1e-12);
    EXPECT_NEAR(x.y(), 0.5, 1e-12);
    EXPECT_NEAR(x.z(), 0.5, 1e-12);
}

TEST_F(HexahedronTransformTest, JacobianUnitCube)
{
    // Reference cube is [-1,1]^3 with volume = 8
    // Physical unit cube is [0,1]^3 with volume = 1
    // Jacobian: dx/dxi = 0.5 for each component
    // J = 0.5 * I, so detJ = 0.5^3 = 0.125
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->setIntegrationPoint(xi);

    EXPECT_NEAR(transform_->detJ(), 0.125, 1e-12);
}

// =============================================================================
// Triangle Boundary Transform Tests (using ElementTransform)
// =============================================================================

class TriangleFacetTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createSingleTriBdrMesh();
        transform_ = std::make_unique<ElementTransform>();
        bindElementToTransform(*transform_, mesh_, 0, true);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(TriangleFacetTransformTest, GeometryInfo)
{
    EXPECT_EQ(transform_->geometry(), Geometry::Triangle);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numNodes(), 3);
}

TEST_F(TriangleFacetTransformTest, TransformVertices)
{
    // Test transformation of vertices
    Vector3 xi(0.0, 0.0, 0.0); // Vertex 0
    Vector3 x;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    xi << 1.0, 0.0, 0.0; // Vertex 1
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    xi << 0.0, 1.0, 0.0; // Vertex 2
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

TEST_F(TriangleFacetTransformTest, JacobianDeterminant)
{
    // Reference triangle has vertices (0,0), (1,0), (0,1), area = 0.5
    // Physical unit triangle has the same vertices, area = 0.5
    // detJ = physical_area / reference_area = 1
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->setIntegrationPoint(xi);

    EXPECT_NEAR(transform_->detJ(), 1.0, 1e-12);
}

TEST_F(TriangleFacetTransformTest, NormalVector)
{
    // The triangle lies in the xy-plane, so normal should be along z-axis
    Vector3 xi(0.33, 0.33, 0.0);
    transform_->setIntegrationPoint(xi);

    Vector3 n = transform_->normal();
    EXPECT_NEAR(std::abs(n.z()), 1.0, 1e-12);
}

// =============================================================================
// Square Boundary Transform Tests (using ElementTransform)
// =============================================================================

class SquareFacetTransformTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createSingleSquareBdrMesh();
        transform_ = std::make_unique<ElementTransform>();
        bindElementToTransform(*transform_, mesh_, 0, true);
    }

    Mesh mesh_;
    std::unique_ptr<ElementTransform> transform_;
};

TEST_F(SquareFacetTransformTest, GeometryInfo)
{
    EXPECT_EQ(transform_->geometry(), Geometry::Square);
    EXPECT_EQ(transform_->dim(), 2);
    EXPECT_EQ(transform_->numNodes(), 4);
}

TEST_F(SquareFacetTransformTest, TransformCorners)
{
    // Reference square: [-1, 1]^2

    // Corner (-1, -1) -> (0, 0)
    Vector3 xi(-1.0, -1.0, 0.0);
    Vector3 x;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 0.0, 1e-12);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);

    // Corner (1, 1) -> (1, 1)
    xi << 1.0, 1.0, 0.0;
    x = transform_->transform(xi);
    EXPECT_NEAR(x.x(), 1.0, 1e-12);
    EXPECT_NEAR(x.y(), 1.0, 1e-12);
}

TEST_F(SquareFacetTransformTest, NormalVector)
{
    // The square lies in the xy-plane, so normal should be along z-axis
    Vector3 xi(0.0, 0.0, 0.0);
    transform_->setIntegrationPoint(xi);

    Vector3 n = transform_->normal();
    EXPECT_NEAR(std::abs(n.z()), 1.0, 1e-12);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(IntegrationTransformTest, IntegrateOverTetrahedron)
{
    // Integrate f(x,y,z) = 1 over unit tetrahedron
    // Result should be volume = 1/6

    Mesh mesh = createSingleTetMesh();
    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0);

    auto rule = quadrature::getTetrahedron(2);
    Real integral = 0.0;

    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        integral += ip.weight * trans.weight();
    }

    EXPECT_NEAR(integral, 1.0 / 6.0, 1e-12);
}

TEST(IntegrationTransformTest, IntegrateOverTriangleBoundary)
{
    // Integrate f(x,y) = 1 over unit triangle
    // Result should be area = 0.5

    Mesh mesh = createSingleTriBdrMesh();
    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0, true);

    auto rule = quadrature::getTriangle(2);
    Real integral = 0.0;

    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        integral += ip.weight * trans.weight();
    }

    EXPECT_NEAR(integral, 0.5, 1e-12);
}

// =============================================================================
// Scaled Element Tests
// =============================================================================

TEST(ScaledElementTest, ScaledTetrahedron)
{
    // Create a tetrahedron scaled by factor of 2
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(2.0, 0.0, 0.0);
    mesh.addVertex(0.0, 2.0, 0.0);
    mesh.addVertex(0.0, 0.0, 2.0);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});

    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0);

    Vector3 xi(0.0, 0.0, 0.0);
    trans.setIntegrationPoint(xi);

    // Reference tet volume = 1/6
    // Physical tet volume = (2^3) * (1/6) = 8/6 = 4/3
    // detJ = physical_vol / reference_vol = 8
    EXPECT_NEAR(trans.detJ(), 8.0, 1e-12);
    EXPECT_NEAR(trans.weight(), 8.0, 1e-12);
}

TEST(ScaledElementTest, ScaledTriangleBoundary)
{
    // Create a triangle scaled by factor of 2
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(2.0, 0.0, 0.0);
    mesh.addVertex(0.0, 2.0, 0.0);
    mesh.addBdrElement(Geometry::Triangle, {0, 1, 2});

    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0, true);

    Vector3 xi(0.0, 0.0, 0.0);
    trans.setIntegrationPoint(xi);

    // Reference tri area = 0.5
    // Physical tri area = (2^2) * 0.5 = 2.0
    // detJ = physical_area / reference_area = 4
    EXPECT_NEAR(trans.detJ(), 4.0, 1e-12);
    EXPECT_NEAR(trans.weight(), 4.0, 1e-12);
}

// =============================================================================
// Gradient Transformation Verification
// =============================================================================

TEST(GradientTransformTest, NumericalVerification)
{
    // Verify gradient transformation using numerical differentiation

    // Create a non-trivial tetrahedron
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(1.0, 0.0, 0.0);
    mesh.addVertex(0.3, 0.8, 0.1);
    mesh.addVertex(0.2, 0.1, 0.9);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});

    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0);

    // Set integration point
    Vector3 xi(0.2, 0.3, 0.1);
    trans.setIntegrationPoint(xi);

    // For any linear element, the gradient transformation should be exact
    // Test that J^{-T} * J^{T} = I

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

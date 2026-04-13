#include "core/geometry.hpp"
#include "fe/fe_collection.hpp"
#include "fe/quadrature.hpp"
#include "field/fe_space.hpp"
#include "mesh/mesh.hpp"
#include <gtest/gtest.h>


using namespace mpfem;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a simple 2D triangular mesh
Mesh createTriMesh2D()
{
    Mesh mesh(2, 4, 2);

    // Two triangles sharing an edge
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2
    mesh.addVertex(1.0, 1.0, 0.0); // 3

    mesh.addElement(Geometry::Triangle, {0, 1, 2}); // Element 0
    mesh.addElement(Geometry::Triangle, {1, 3, 2}); // Element 1

    // Boundary edges
    mesh.addBdrElement(Geometry::Segment, {0, 1}); // Bottom
    mesh.addBdrElement(Geometry::Segment, {1, 3}); // Right

    mesh.buildTopology();

    return mesh;
}

/// Create a simple 3D tetrahedral mesh
Mesh createTetMesh3D()
{
    Mesh mesh(3, 5, 4);

    // Two tetrahedra sharing a face
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 0.0, 1.0); // 3
    mesh.addVertex(1.0, 1.0, 1.0); // 4

    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}); // Element 0
    mesh.addElement(Geometry::Tetrahedron, {1, 2, 3, 4}); // Element 1

    // Boundary faces (triangles)
    mesh.addBdrElement(Geometry::Triangle, {0, 1, 2});
    mesh.addBdrElement(Geometry::Triangle, {0, 1, 3});
    mesh.addBdrElement(Geometry::Triangle, {0, 2, 3});
    mesh.addBdrElement(Geometry::Triangle, {1, 2, 4});

    mesh.buildTopology();

    return mesh;
}

/// Create a simple 2D quadrilateral mesh
Mesh createQuadMesh2D()
{
    Mesh mesh(2, 9, 4);

    // 2x2 grid of quads
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(2.0, 0.0, 0.0); // 2
    mesh.addVertex(0.0, 1.0, 0.0); // 3
    mesh.addVertex(1.0, 1.0, 0.0); // 4
    mesh.addVertex(2.0, 1.0, 0.0); // 5
    mesh.addVertex(0.0, 2.0, 0.0); // 6
    mesh.addVertex(1.0, 2.0, 0.0); // 7
    mesh.addVertex(2.0, 2.0, 0.0); // 8

    // Quads (counter-clockwise)
    mesh.addElement(Geometry::Square, {0, 1, 4, 3}); // Element 0
    mesh.addElement(Geometry::Square, {1, 2, 5, 4}); // Element 1
    mesh.addElement(Geometry::Square, {3, 4, 7, 6}); // Element 2
    mesh.addElement(Geometry::Square, {4, 5, 8, 7}); // Element 3

    // Boundary edges
    mesh.addBdrElement(Geometry::Segment, {0, 1});
    mesh.addBdrElement(Geometry::Segment, {1, 2});
    mesh.addBdrElement(Geometry::Segment, {2, 5});
    mesh.addBdrElement(Geometry::Segment, {5, 8});

    mesh.buildTopology();

    return mesh;
}

/// Create a mixed 2D mesh: one quad + one triangle sharing an edge
Mesh createMixedMesh2D()
{
    Mesh mesh(2, 6, 2);

    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(1.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 1.0, 0.0); // 3
    mesh.addVertex(2.0, 0.0, 0.0); // 4
    mesh.addVertex(2.0, 1.0, 0.0); // 5

    mesh.addElement(Geometry::Square, {0, 1, 2, 3});
    mesh.addElement(Geometry::Triangle, {1, 4, 2});

    mesh.addBdrElement(Geometry::Segment, {0, 1});
    mesh.addBdrElement(Geometry::Segment, {1, 4});
    mesh.addBdrElement(Geometry::Segment, {4, 2});
    mesh.addBdrElement(Geometry::Segment, {2, 3});
    mesh.addBdrElement(Geometry::Segment, {3, 0});

    mesh.buildTopology();
    return mesh;
}

/// Create a 3D hexahedral mesh
Mesh createHexMesh3D()
{
    Mesh mesh(3, 8, 0);

    // Single cube
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(1.0, 1.0, 0.0); // 2
    mesh.addVertex(0.0, 1.0, 0.0); // 3
    mesh.addVertex(0.0, 0.0, 1.0); // 4
    mesh.addVertex(1.0, 0.0, 1.0); // 5
    mesh.addVertex(1.0, 1.0, 1.0); // 6
    mesh.addVertex(0.0, 1.0, 1.0); // 7

    mesh.addElement(Geometry::Cube, {0, 1, 2, 3, 4, 5, 6, 7});

    mesh.buildTopology();

    return mesh;
}

std::vector<Index> getElementDofsVec(const FESpace& fes, Index elemIdx)
{
    std::vector<Index> dofs(static_cast<size_t>(fes.numElementDofs(elemIdx)), InvalidIndex);
    if (!dofs.empty()) {
        fes.getElementDofs(elemIdx, std::span<Index> {dofs});
    }
    return dofs;
}

std::vector<Index> getBdrElementDofsVec(const FESpace& fes, Index bdrIdx)
{
    std::vector<Index> dofs(static_cast<size_t>(fes.numBdrElementDofs(bdrIdx)), InvalidIndex);
    if (!dofs.empty()) {
        fes.getBdrElementDofs(bdrIdx, std::span<Index> {dofs});
    }
    return dofs;
}

// =============================================================================
// Linear FE Space Tests (Order 1)
// =============================================================================

class FESpaceLinearTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTriMesh2D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceLinearTest, BasicProperties)
{
    EXPECT_EQ(feSpace_->order(), 1);
    EXPECT_EQ(feSpace_->vdim(), 1);
    EXPECT_EQ(feSpace_->dim(), 2);
}

TEST_F(FESpaceLinearTest, NumDofs)
{
    // Linear elements: one DOF per vertex
    // 4 vertices = 4 DOFs
    EXPECT_EQ(feSpace_->numDofs(), 4);
}

TEST_F(FESpaceLinearTest, ElementDofs)
{
    // Check DOF mapping for each element
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs0.size(), 3); // 3 nodes per triangle

    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);
    EXPECT_EQ(dofs1.size(), 3);
}

TEST_F(FESpaceLinearTest, DofMappingConsistency)
{
    // Element 0: vertices (0, 1, 2) -> DOFs (0, 1, 2)
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs0[0], 0);
    EXPECT_EQ(dofs0[1], 1);
    EXPECT_EQ(dofs0[2], 2);

    // Element 1: vertices (1, 3, 2) -> DOFs (1, 3, 2)
    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);
    EXPECT_EQ(dofs1[0], 1);
    EXPECT_EQ(dofs1[1], 3);
    EXPECT_EQ(dofs1[2], 2);
}

TEST_F(FESpaceLinearTest, SharedDofConsistency)
{
    // Vertex 2 is shared by both elements
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);

    // Vertex 2 -> DOF 2 in both elements
    EXPECT_EQ(dofs0[2], dofs1[2]);
    EXPECT_EQ(dofs0[2], 2);
}

TEST_F(FESpaceLinearTest, BoundaryElementDofs)
{
    // Boundary 0: vertices (0, 1) -> DOFs (0, 1)
    std::vector<Index> dofs = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs.size(), 3);
}

// =============================================================================
// Quadratic FE Space Tests (Order 2)
// =============================================================================

class FESpaceQuadraticTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTriMesh2D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(2));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceQuadraticTest, BasicProperties)
{
    EXPECT_EQ(feSpace_->order(), 2);
    EXPECT_EQ(feSpace_->vdim(), 1);
}

TEST_F(FESpaceQuadraticTest, NumDofs)
{
    // 2D quadratic triangle on this mesh: 4 corner vertex DOFs + 5 edge DOFs.
    EXPECT_EQ(feSpace_->numDofs(), 9);
}

TEST_F(FESpaceQuadraticTest, ElementDofs)
{
    // Quadratic triangle has 6 local DOFs: 3 vertex + 3 edge.
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs0.size(), 6); // Reference element has 6 DOFs
    for (Index d : dofs0) {
        EXPECT_NE(d, InvalidIndex);
    }

    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);
    EXPECT_EQ(dofs1.size(), 6);
}

TEST_F(FESpaceQuadraticTest, EdgeDofSharing)
{
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);

    // Shared vertices.
    EXPECT_EQ(dofs0[1], dofs1[0]) << "Vertex 1 should have same DOF in both elements";
    EXPECT_EQ(dofs0[2], dofs1[2]) << "Vertex 2 should have same DOF in both elements";

    // Shared edge (1,2): element0 local edge2, element1 local edge1.
    EXPECT_EQ(dofs0[5], dofs1[4]) << "Shared edge should have same global DOF";
}

// =============================================================================
// 3D Tetrahedral FE Space Tests
// =============================================================================

class FESpaceTetTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTetMesh3D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceTetTest, NumDofs)
{
    // 5 vertices = 5 DOFs for linear elements
    EXPECT_EQ(feSpace_->numDofs(), 5);
}

TEST_F(FESpaceTetTest, ElementDofs)
{
    // Linear tetrahedron has 4 DOFs
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs0.size(), 4);

    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);
    EXPECT_EQ(dofs1.size(), 4);
}

TEST_F(FESpaceTetTest, SharedFaceDofs)
{
    // Elements 0 and 1 share face with vertices (1, 2, 3)
    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);

    // Element 0: vertices (0, 1, 2, 3)
    // Element 1: vertices (1, 2, 3, 4)
    // Shared vertices: 1, 2, 3

    // DOF 1 in element 0 should equal DOF 0 in element 1 (vertex 1)
    EXPECT_EQ(dofs0[1], dofs1[0]);
    // DOF 2 in element 0 should equal DOF 1 in element 1 (vertex 2)
    EXPECT_EQ(dofs0[2], dofs1[1]);
    // DOF 3 in element 0 should equal DOF 2 in element 1 (vertex 3)
    EXPECT_EQ(dofs0[3], dofs1[2]);
}

// =============================================================================
// Quadrilateral FE Space Tests
// =============================================================================

class FESpaceQuadTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createQuadMesh2D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceQuadTest, NumDofs)
{
    // 9 vertices = 9 DOFs
    EXPECT_EQ(feSpace_->numDofs(), 9);
}

TEST_F(FESpaceQuadTest, ElementDofs)
{
    // Linear quad has 4 DOFs
    std::vector<Index> dofs = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs.size(), 4);
}

TEST_F(FESpaceQuadTest, SharedEdgeDofs)
{
    // Element 0: vertices {0, 1, 4, 3}
    // Element 1: vertices {1, 2, 5, 4}
    // Shared edge: vertices {1, 4}

    std::vector<Index> dofs0 = getElementDofsVec(*feSpace_, 0);
    std::vector<Index> dofs1 = getElementDofsVec(*feSpace_, 1);

    // Element 0 DOFs: [vertex0, vertex1, vertex4, vertex3] = [0, 1, 4, 3]
    // Element 1 DOFs: [vertex1, vertex2, vertex5, vertex4] = [1, 2, 5, 4]

    // Vertex 1: DOF index 1 in elem 0, DOF index 0 in elem 1
    EXPECT_EQ(dofs0[1], dofs1[0]);
    // Vertex 4: DOF index 2 in elem 0, DOF index 3 in elem 1
    EXPECT_EQ(dofs0[2], dofs1[3]);
}

// =============================================================================
// Quadratic Quadrilateral Tests
// =============================================================================

class FESpaceQuadQuadraticTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createQuadMesh2D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(2));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceQuadQuadraticTest, NumDofs)
{
    // 2D quadratic quad: 9 vertex DOFs + 12 edge DOFs + 4 cell-interior DOFs.
    EXPECT_EQ(feSpace_->numDofs(), 25);
}

TEST_F(FESpaceQuadQuadraticTest, ElementDofs)
{
    // Quadratic quad has 9 local DOFs and all should map to valid globals.
    std::vector<Index> dofs = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs.size(), 9); // Reference element has 9 DOFs
    for (int i = 0; i < 9; ++i) {
        EXPECT_NE(dofs[i], InvalidIndex) << "DOF " << i << " should be valid";
    }
}

// =============================================================================
// Mixed Geometry FE Space Tests
// =============================================================================

class FESpaceMixedGeometryTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createMixedMesh2D();
    }

    Mesh mesh_;
};

TEST_F(FESpaceMixedGeometryTest, LinearMixedMeshDofs)
{
    FESpace fes(&mesh_, std::make_unique<H1Collection>(1));

    EXPECT_EQ(fes.numDofs(), 5);

    std::vector<Index> quadDofs = getElementDofsVec(fes, 0);
    std::vector<Index> triDofs = getElementDofsVec(fes, 1);
    EXPECT_EQ(quadDofs.size(), 4);
    EXPECT_EQ(triDofs.size(), 3);

    // Shared edge vertices (1,2) must map to same global DOFs.
    EXPECT_EQ(quadDofs[1], triDofs[0]);
    EXPECT_EQ(quadDofs[2], triDofs[2]);
}

TEST_F(FESpaceMixedGeometryTest, QuadraticMixedMeshDofs)
{
    FESpace fes(&mesh_, std::make_unique<H1Collection>(2));

    // 5 used vertices + 6 edge + 1 quad cell interior = 12 scalar DOFs.
    EXPECT_EQ(fes.numDofs(), 12);

    std::vector<Index> quadDofs = getElementDofsVec(fes, 0);
    std::vector<Index> triDofs = getElementDofsVec(fes, 1);
    EXPECT_EQ(quadDofs.size(), 9);
    EXPECT_EQ(triDofs.size(), 6);

    // Shared edge midpoint should be shared across both elements.
    EXPECT_EQ(quadDofs[5], triDofs[4]);
}

// =============================================================================
// Vector Field Tests (vdim > 1)
// =============================================================================

class FESpaceVectorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTriMesh2D();
        // Create a vector field with 2 components
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1, 2));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceVectorTest, VectorDimension)
{
    EXPECT_EQ(feSpace_->vdim(), 2);
}

TEST_F(FESpaceVectorTest, NumDofs)
{
    // 4 vertices * 2 components = 8 DOFs
    EXPECT_EQ(feSpace_->numDofs(), 8);
}

TEST_F(FESpaceVectorTest, ElementDofs)
{
    // 3 nodes * 2 components = 6 DOFs per element
    std::vector<Index> dofs = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs.size(), 6);
}

// =============================================================================
// 3D Vector Field Tests
// =============================================================================

class FESpace3DVectorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTetMesh3D();
        // Create a 3D displacement field
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1, 3));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpace3DVectorTest, VectorDimension)
{
    EXPECT_EQ(feSpace_->vdim(), 3);
}

TEST_F(FESpace3DVectorTest, NumDofs)
{
    // 5 vertices * 3 components = 15 DOFs
    EXPECT_EQ(feSpace_->numDofs(), 15);
}

TEST_F(FESpace3DVectorTest, ElementDofs)
{
    // 4 nodes * 3 components = 12 DOFs per element
    std::vector<Index> dofs = getElementDofsVec(*feSpace_, 0);
    EXPECT_EQ(dofs.size(), 12);
}

// =============================================================================
// FE Collection Tests
// =============================================================================

TEST(FECollectionTest, LinearCollection)
{
    H1Collection fec(1);

    EXPECT_EQ(fec.order(), 1);

    // Should have shape functions for all geometry types
    EXPECT_NE(fec.get(Geometry::Segment), nullptr);
    EXPECT_NE(fec.get(Geometry::Triangle), nullptr);
    EXPECT_NE(fec.get(Geometry::Square), nullptr);
    EXPECT_NE(fec.get(Geometry::Tetrahedron), nullptr);
    EXPECT_NE(fec.get(Geometry::Cube), nullptr);
}

TEST(FECollectionTest, QuadraticCollection)
{
    H1Collection fec(2);

    EXPECT_EQ(fec.order(), 2);

    // Check numDofs for each geometry
    EXPECT_EQ(fec.get(Geometry::Segment)->numDofs(), 3);
    EXPECT_EQ(fec.get(Geometry::Triangle)->numDofs(), 6);
    EXPECT_EQ(fec.get(Geometry::Square)->numDofs(), 9);
    EXPECT_EQ(fec.get(Geometry::Tetrahedron)->numDofs(), 10);
    EXPECT_EQ(fec.get(Geometry::Cube)->numDofs(), 27);
}

// =============================================================================
// Boundary Element DOF Tests
// =============================================================================

class FESpaceBdrTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createTriMesh2D();
        feSpace_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(1));
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> feSpace_;
};

TEST_F(FESpaceBdrTest, BoundaryElementDofs)
{
    // Get DOFs for boundary element 0 (edge 0-1)
    std::vector<Index> dofs = getBdrElementDofsVec(*feSpace_, 0);

    // Linear edge has 2 DOFs
    EXPECT_EQ(dofs.size(), 2);

    // Should be vertex DOFs 0 and 1
    EXPECT_EQ(dofs[0], 0);
    EXPECT_EQ(dofs[1], 1);
}

TEST_F(FESpaceBdrTest, NumBdrElementDofs)
{
    EXPECT_EQ(feSpace_->numBdrElementDofs(0), 2);
}

// =============================================================================
// Reference Element Access Tests
// =============================================================================

TEST_F(FESpaceLinearTest, ReferenceElementAccess)
{
    // Get reference element for triangle
    const ReferenceElement* refElem = feSpace_->elementRefElement(0);
    ASSERT_NE(refElem, nullptr);

    EXPECT_EQ(refElem->geometry(), Geometry::Triangle);
    EXPECT_EQ(refElem->numDofs(), 3);
}
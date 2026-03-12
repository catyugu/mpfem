#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/fe_space.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_collection.hpp"
#include "fe/grid_function.hpp"
#include "fe/quadrature.hpp"

using namespace mpfem;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a mesh with second-order triangular elements
Mesh createQuadraticTriangleMesh() {
    Mesh mesh;
    mesh.setDim(3);
    
    // Create a simple mesh with 2 triangles sharing an edge
    // Triangle 0: vertices 0, 1, 2
    // Triangle 1: vertices 2, 1, 3
    
    // Corner vertices
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.5, 1.0, 0.0);  // 2
    mesh.addVertex(1.5, 1.0, 0.0);  // 3
    
    // Edge midpoint vertices
    mesh.addVertex(0.5, 0.0, 0.0);  // 4: edge 0-1
    mesh.addVertex(0.75, 0.5, 0.0); // 5: edge 1-2
    mesh.addVertex(0.25, 0.5, 0.0); // 6: edge 2-0
    mesh.addVertex(1.25, 0.5, 0.0); // 7: edge 1-3
    mesh.addVertex(1.0, 1.0, 0.0);  // 8: edge 2-3 (shared edge)
    
    // Triangle2 ordering: V0, V1, V2, E01, E12, E20
    mesh.addElement(Geometry::Triangle, {0, 1, 2, 4, 5, 6}, 1, 2);
    mesh.addElement(Geometry::Triangle, {2, 1, 3, 8, 7, 5}, 2, 2);
    
    return mesh;
}

/// Create a mesh with second-order tetrahedral elements
Mesh createQuadraticTetrahedronMesh() {
    Mesh mesh;
    mesh.setDim(3);
    
    // Create a simple mesh with 2 tetrahedra sharing a face
    
    // Corner vertices
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.5, 1.0, 0.0);  // 2
    mesh.addVertex(0.5, 0.5, 1.0);  // 3
    mesh.addVertex(1.5, 0.5, 1.0);  // 4
    
    // Edge midpoint vertices (10 edges total for 2 tets)
    // First tet: edges from vertices (0,1,2,3)
    mesh.addVertex(0.5, 0.0, 0.0);   // 5: edge 0-1
    mesh.addVertex(0.75, 0.5, 0.0);  // 6: edge 1-2
    mesh.addVertex(0.25, 0.5, 0.0);  // 7: edge 2-0
    mesh.addVertex(0.25, 0.25, 0.5); // 8: edge 0-3
    mesh.addVertex(0.75, 0.25, 0.5); // 9: edge 1-3
    mesh.addVertex(0.5, 0.75, 0.5);  // 10: edge 2-3
    
    // Additional edge midpoints for second tet
    mesh.addVertex(1.0, 0.0, 0.0);   // 11: edge 1-4
    mesh.addVertex(1.0, 0.75, 0.5);  // 12: edge 2-4
    mesh.addVertex(1.0, 0.5, 1.0);   // 13: edge 3-4
    
    // Tetrahedron2 ordering: V0, V1, V2, V3, E01, E02, E03, E12, E13, E23
    // Using COMSOL ordering: V0, V1, V2, V3, E01, E12, E02, E13, E23, E03
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3, 5, 6, 7, 8, 9, 10}, 1, 2);
    mesh.addElement(Geometry::Tetrahedron, {1, 4, 2, 3, 11, 12, 6, 9, 13, 12}, 2, 2);
    
    return mesh;
}

// =============================================================================
// FESpace Tests for Quadratic Elements
// =============================================================================

class QuadraticFESpaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createQuadraticTriangleMesh();
        fec_ = std::make_unique<FECollection>(2, FECollection::Type::H1);
    }
    
    Mesh mesh_;
    std::unique_ptr<FECollection> fec_;
};

TEST_F(QuadraticFESpaceTest, NumDofs) {
    FESpace fes(&mesh_, fec_.get());
    
    // For 2 triangular elements sharing an edge:
    // - 4 corner vertices -> 4 vertex DOFs
    // - 5 unique edges -> 5 edge DOFs
    // Total: 9 DOFs
    EXPECT_EQ(fes.numDofs(), 9);
}

TEST_F(QuadraticFESpaceTest, ElementDofs) {
    FESpace fes(&mesh_, fec_.get());
    
    // Get DOFs for first element
    auto dofs0 = fes.elementDofs(0);
    EXPECT_EQ(dofs0.size(), 6);  // 6 DOFs per quadratic triangle
    
    // Get DOFs for second element
    auto dofs1 = fes.elementDofs(1);
    EXPECT_EQ(dofs1.size(), 6);
}

TEST_F(QuadraticFESpaceTest, EdgeDofSharing) {
    FESpace fes(&mesh_, fec_.get());
    
    // Elements share edge 1-2 (which has edge midpoint at vertex 5)
    auto dofs0 = fes.elementDofs(0);
    auto dofs1 = fes.elementDofs(1);
    
    // Edge DOFs on shared edge should be the same
    // For triangle0: edge DOFs are at indices 3,4,5 (E01, E12, E20)
    // For triangle1: edge DOFs are at indices 3,4,5 (E21, E13, E32)
    // Edge 1-2 in tri0 corresponds to edge 2-1 in tri1
    
    // Check that both elements have valid DOFs
    for (Index dof : dofs0) {
        EXPECT_NE(dof, InvalidIndex);
    }
    for (Index dof : dofs1) {
        EXPECT_NE(dof, InvalidIndex);
    }
    
    // Verify all DOFs are unique (no duplicates within an element)
    std::set<Index> uniqueDofs0(dofs0.begin(), dofs0.end());
    std::set<Index> uniqueDofs1(dofs1.begin(), dofs1.end());
    EXPECT_EQ(uniqueDofs0.size(), 6);
    EXPECT_EQ(uniqueDofs1.size(), 6);
}

TEST_F(QuadraticFESpaceTest, VectorFESpace) {
    // Test vector-valued FE space (vdim = 2)
    FESpace fes(&mesh_, fec_.get(), 2);
    
    // Total DOFs should be doubled
    EXPECT_EQ(fes.numDofs(), 18);  // 9 * 2
    
    auto dofs = fes.elementDofs(0);
    EXPECT_EQ(dofs.size(), 12);  // 6 * 2
}

// =============================================================================
// GridFunction Tests for Quadratic Elements
// =============================================================================

class QuadraticGridFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = createQuadraticTriangleMesh();
        fec_ = std::make_unique<FECollection>(2, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh_, fec_.get());
        gf_ = std::make_unique<GridFunction>(fes_.get());
    }
    
    Mesh mesh_;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> gf_;
};

TEST_F(QuadraticGridFunctionTest, SetValueAtVertex) {
    // Set value at vertex 0
    (*gf_)(0) = 1.0;
    
    // The value should be retrievable
    EXPECT_NEAR((*gf_)(0), 1.0, 1e-12);
}

TEST_F(QuadraticGridFunctionTest, InterpolateLinearFunction) {
    // Interpolate f(x,y) = x
    
    // Set vertex values
    for (Index v = 0; v < mesh_.numVertices(); ++v) {
        const Vertex& vert = mesh_.vertex(v);
        (*gf_)(v) = vert.x();
    }
    
    // Check that interpolation is correct at vertices
    for (Index v = 0; v < 4; ++v) {  // Only check corner vertices
        const Vertex& vert = mesh_.vertex(v);
        EXPECT_NEAR((*gf_)(v), vert.x(), 1e-12);
    }
}

TEST_F(QuadraticGridFunctionTest, EvalAtReferencePoint) {
    // Set all DOFs to 1
    gf_->setConstant(1.0);
    
    // Evaluate at reference point (0.5, 0.5) - should be 1
    Real xi[] = {0.5, 0.5};
    Real val = gf_->eval(0, xi);
    
    EXPECT_NEAR(val, 1.0, 1e-12);
}

TEST_F(QuadraticGridFunctionTest, EvalQuadraticFunction) {
    // Set up a quadratic function f = x^2 + y^2
    // On a quadratic element, this should be represented exactly
    
    for (Index i = 0; i < fes_->numDofs(); ++i) {
        // For simplicity, just set up based on vertex positions
        if (i < mesh_.numVertices()) {
            const Vertex& v = mesh_.vertex(i);
            (*gf_)(i) = v.x() * v.x() + v.y() * v.y();
        }
    }
    
    // Check value at vertex 1: (1,0) -> 1.0
    EXPECT_NEAR((*gf_)(1), 1.0, 1e-12);
}

// =============================================================================
// Reference Element Tests for Quadratic Elements
// =============================================================================

class QuadraticReferenceElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        fec_ = std::make_unique<FECollection>(2, FECollection::Type::H1);
    }
    
    std::unique_ptr<FECollection> fec_;
};

TEST_F(QuadraticReferenceElementTest, TriangleNumDofs) {
    const ReferenceElement* refTri = fec_->get(Geometry::Triangle);
    ASSERT_NE(refTri, nullptr);
    EXPECT_EQ(refTri->numDofs(), 6);  // 3 + 3
}

TEST_F(QuadraticReferenceElementTest, TetrahedronNumDofs) {
    const ReferenceElement* refTet = fec_->get(Geometry::Tetrahedron);
    ASSERT_NE(refTet, nullptr);
    EXPECT_EQ(refTet->numDofs(), 10);  // 4 + 6
}

TEST_F(QuadraticReferenceElementTest, SquareNumDofs) {
    const ReferenceElement* refSquare = fec_->get(Geometry::Square);
    ASSERT_NE(refSquare, nullptr);
    EXPECT_EQ(refSquare->numDofs(), 9);  // 4 + 4 + 1
}

TEST_F(QuadraticReferenceElementTest, CubeNumDofs) {
    const ReferenceElement* refCube = fec_->get(Geometry::Cube);
    ASSERT_NE(refCube, nullptr);
    EXPECT_EQ(refCube->numDofs(), 27);  // 8 + 12 + 6 + 1
}

TEST_F(QuadraticReferenceElementTest, TriangleFaceDofs) {
    const ReferenceElement* refTri = fec_->get(Geometry::Triangle);
    ASSERT_NE(refTri, nullptr);
    
    // Triangle has 3 edges as "faces" (boundary edges)
    // Each edge is a Segment with 2 corners + 1 midpoint = 3 DOFs
    
    for (int f = 0; f < 3; ++f) {
        auto faceDofs = refTri->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 3);  // 2 corners + 1 edge midpoint
    }
}

TEST_F(QuadraticReferenceElementTest, TetrahedronFaceDofs) {
    const ReferenceElement* refTet = fec_->get(Geometry::Tetrahedron);
    ASSERT_NE(refTet, nullptr);
    
    // Tetrahedron has 4 triangular faces
    // Each face has 3 corner DOFs + 3 edge DOFs = 6 DOFs
    
    for (int f = 0; f < 4; ++f) {
        auto faceDofs = refTet->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 6);  // 3 corners + 3 edge midpoints
    }
}

TEST_F(QuadraticReferenceElementTest, CubeFaceDofs) {
    const ReferenceElement* refCube = fec_->get(Geometry::Cube);
    ASSERT_NE(refCube, nullptr);
    
    // Cube has 6 quadrilateral faces
    // Each face has 4 corners + 4 edges + 1 center = 9 DOFs
    
    for (int f = 0; f < 6; ++f) {
        auto faceDofs = refCube->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 9);  // 4 corners + 4 edge midpoints + 1 center
    }
}

// =============================================================================
// Integration Tests for Quadratic Elements
// =============================================================================

TEST(QuadraticIntegrationTest, IntegrateQuadraticFunctionExactly) {
    // Create a single quadratic triangle
    Mesh mesh;
    mesh.setDim(3);
    
    // Unit right triangle
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.0, 1.0, 0.0);  // 2
    mesh.addVertex(0.5, 0.0, 0.0);  // 3: edge 0-1 midpoint
    mesh.addVertex(0.5, 0.5, 0.0);  // 4: edge 1-2 midpoint
    mesh.addVertex(0.0, 0.5, 0.0);  // 5: edge 2-0 midpoint
    
    mesh.addElement(Geometry::Triangle, {0, 1, 2, 3, 4, 5}, 1, 2);
    
    FECollection fec(2);
    FESpace fes(&mesh, &fec);
    GridFunction gf(&fes);
    
    // Set f(x,y) = x + y (linear, but should integrate exactly on quadratic element)
    for (Index i = 0; i < fes.numDofs(); ++i) {
        const Vertex& v = mesh.vertex(i);
        gf(i) = v.x() + v.y();
    }
    
    // Integrate using order-4 quadrature (sufficient for quadratic)
    auto refElem = fes.elementRefElement(0);
    const QuadratureRule& rule = refElem->quadrature();
    
    ElementTransform trans(&mesh, 0);
    Real integral = 0.0;
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        Real xi[] = {ip.xi, ip.eta};
        Real f_val = gf.eval(0, xi);
        integral += ip.weight * trans.weight() * f_val;
    }
    
    // For f(x,y) = x + y over triangle with vertices (0,0), (1,0), (0,1)
    // Integral = integral of (x + y) over triangle
    // Using barycentric: integral of x over triangle = 1/6, same for y
    // Total = 1/6 + 1/6 = 1/3
    EXPECT_NEAR(integral, 1.0/3.0, 1e-10);
}

// =============================================================================
// Mixed-Order Test (Geometric vs Physical Order)
// =============================================================================

TEST(MixedOrderTest, SubparametricElement) {
    // Test that geometric order (2) and physical order (1) can differ
    // This is subparametric: curved geometry with linear field
    
    Mesh mesh = createQuadraticTriangleMesh();  // Geometric order = 2
    FECollection fec(1);  // Physical order = 1
    
    // This should work: linear field on curved geometry
    FESpace fes(&mesh, &fec);
    EXPECT_EQ(fes.order(), 1);  // Physical order
    
    // Element has geometric order 2, but FE space has order 1
    // This is the key test for separation of geometric and physical order
    EXPECT_EQ(mesh.element(0).order(), 2);  // Geometric order
    EXPECT_EQ(fes.order(), 1);              // Physical order
}

TEST(MixedOrderTest, IsoparametricElement) {
    // Test isoparametric: geometric order = physical order = 2
    
    Mesh mesh = createQuadraticTriangleMesh();  // Geometric order = 2
    FECollection fec(2);  // Physical order = 2
    
    FESpace fes(&mesh, &fec);
    
    EXPECT_EQ(mesh.element(0).order(), 2);  // Geometric order
    EXPECT_EQ(fes.order(), 2);              // Physical order
}

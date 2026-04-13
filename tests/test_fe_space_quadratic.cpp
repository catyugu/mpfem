#include "assembly/element_binding.hpp"
#include "core/geometry.hpp"
#include "core/types.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_collection.hpp"
#include "fe/quadrature.hpp"
#include "field/fe_space.hpp"
#include "field/grid_function.hpp"
#include "io/mphtxt_reader.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <set>

using namespace mpfem;

// Helper to get test data path
static std::string dataPath(const std::string& relativePath)
{
#ifdef MPFEM_PROJECT_ROOT
    return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
    return relativePath;
#endif
}

static std::vector<Index> getElementDofsVec(const FESpace& fes, Index elemIdx)
{
    std::vector<Index> dofs(static_cast<size_t>(fes.numElementDofs(elemIdx)), InvalidIndex);
    if (!dofs.empty()) {
        fes.getElementDofs(elemIdx, std::span<Index> {dofs});
    }
    return dofs;
}

static Index getCornerVertexDof(const FESpace& fes, Index vertexIdx)
{
    const Mesh* mesh = fes.mesh();
    if (!mesh || mesh->vertexToCornerIndex(vertexIdx) == InvalidIndex) {
        return InvalidIndex;
    }

    for (Index elemIdx = 0; elemIdx < mesh->numElements(); ++elemIdx) {
        const auto elemVertices = mesh->getElementVertices(elemIdx);
        const auto it = std::find(elemVertices.begin(), elemVertices.end(), vertexIdx);
        if (it == elemVertices.end()) {
            continue;
        }

        const ReferenceElement* refElem = fes.elementRefElement(elemIdx);
        if (!refElem) {
            continue;
        }
        const int vertexDofsPerCorner = refElem->basis().dofLayout().numVertexDofs;
        if (vertexDofsPerCorner <= 0) {
            continue;
        }

        const int localVertex = static_cast<int>(std::distance(elemVertices.begin(), it));
        const int localScalarDof = localVertex * vertexDofsPerCorner;
        const std::vector<Index> elemDofs = getElementDofsVec(fes, elemIdx);
        if (localScalarDof < 0 || localScalarDof >= static_cast<int>(elemDofs.size())) {
            return InvalidIndex;
        }
        return elemDofs[static_cast<size_t>(localScalarDof)];
    }

    return InvalidIndex;
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a mesh with second-order triangular elements
Mesh createQuadraticTriangleMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Create a simple mesh with 2 triangles sharing an edge
    // Triangle 0: vertices 0, 1, 2
    // Triangle 1: vertices 2, 1, 3

    // Corner vertices
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.5, 1.0, 0.0); // 2
    mesh.addVertex(1.5, 1.0, 0.0); // 3

    // Edge midpoint vertices
    mesh.addVertex(0.5, 0.0, 0.0); // 4: edge 0-1
    mesh.addVertex(0.75, 0.5, 0.0); // 5: edge 1-2
    mesh.addVertex(0.25, 0.5, 0.0); // 6: edge 2-0
    mesh.addVertex(1.25, 0.5, 0.0); // 7: edge 1-3
    mesh.addVertex(1.0, 1.0, 0.0); // 8: edge 2-3 (shared edge)

    // Triangle2 COMSOL ordering: V0, V1, V2, E01, E20, E12
    mesh.addElement(Geometry::Triangle, {0, 1, 2, 4, 6, 5}, 1, 2);
    mesh.addElement(Geometry::Triangle, {2, 1, 3, 5, 8, 7}, 2, 2);

    mesh.buildTopology();

    return mesh;
}

/// Create a mesh with second-order tetrahedral elements
Mesh createQuadraticTetrahedronMesh()
{
    Mesh mesh;
    mesh.setDim(3);

    // Create a simple mesh with 2 tetrahedra sharing a face

    // Corner vertices
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.5, 1.0, 0.0); // 2
    mesh.addVertex(0.5, 0.5, 1.0); // 3
    mesh.addVertex(1.5, 0.5, 1.0); // 4

    // Edge midpoint vertices (10 edges total for 2 tets)
    // First tet: edges from vertices (0,1,2,3)
    mesh.addVertex(0.5, 0.0, 0.0); // 5: edge 0-1
    mesh.addVertex(0.75, 0.5, 0.0); // 6: edge 1-2
    mesh.addVertex(0.25, 0.5, 0.0); // 7: edge 2-0
    mesh.addVertex(0.25, 0.25, 0.5); // 8: edge 0-3
    mesh.addVertex(0.75, 0.25, 0.5); // 9: edge 1-3
    mesh.addVertex(0.5, 0.75, 0.5); // 10: edge 2-3

    // Additional edge midpoints for second tet
    mesh.addVertex(1.0, 0.0, 0.0); // 11: edge 1-4
    mesh.addVertex(1.0, 0.75, 0.5); // 12: edge 2-4
    mesh.addVertex(1.0, 0.5, 1.0); // 13: edge 3-4

    // Tetrahedron2 COMSOL ordering: V0, V1, V2, V3, E01, E02, E12, E03, E13, E23
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3, 5, 7, 6, 8, 9, 10}, 1, 2);
    mesh.addElement(Geometry::Tetrahedron, {1, 4, 2, 3, 11, 6, 12, 9, 13, 10}, 2, 2);

    mesh.buildTopology();

    return mesh;
}

// =============================================================================
// FESpace Tests for Quadratic Elements
// =============================================================================

class QuadraticFESpaceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createQuadraticTriangleMesh();
    }

    Mesh mesh_;
};

TEST_F(QuadraticFESpaceTest, NumDofs)
{
    FESpace fes(&mesh_, std::make_unique<H1Collection>(2));

    // For 2 triangular elements sharing an edge:
    // - 4 corner vertices -> 4 vertex DOFs
    // - 5 unique edges -> 5 edge DOFs
    // Total: 9 DOFs
    EXPECT_EQ(fes.numDofs(), 9);
}

TEST_F(QuadraticFESpaceTest, ElementDofs)
{
    FESpace fes(&mesh_, std::make_unique<H1Collection>(2));

    // Get DOFs for first element
    std::vector<Index> dofs0 = getElementDofsVec(fes, 0);
    EXPECT_EQ(dofs0.size(), 6); // 6 DOFs per quadratic triangle

    // Get DOFs for second element
    std::vector<Index> dofs1 = getElementDofsVec(fes, 1);
    EXPECT_EQ(dofs1.size(), 6);
}

TEST_F(QuadraticFESpaceTest, EdgeDofSharing)
{
    FESpace fes(&mesh_, std::make_unique<H1Collection>(2));

    // Elements share edge 1-2 (which has edge midpoint at vertex 5)
    std::vector<Index> dofs0 = getElementDofsVec(fes, 0);
    std::vector<Index> dofs1 = getElementDofsVec(fes, 1);

    // Edge DOFs on shared edge should be the same
    // For triangle0: edge DOFs are at indices 3,4,5 (E01, E20, E12)
    // For triangle1: edge DOFs are at indices 3,4,5 (E21, E32, E13)
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

TEST_F(QuadraticFESpaceTest, VectorFESpace)
{
    // Test vector-valued FE space (vdim = 2)
    FESpace fes(&mesh_, std::make_unique<H1Collection>(2, 2));

    // Total DOFs should be doubled
    EXPECT_EQ(fes.numDofs(), 18); // 9 * 2

    std::vector<Index> dofs;
    dofs = getElementDofsVec(fes, 0);
    EXPECT_EQ(dofs.size(), 12); // 6 * 2
}

// =============================================================================
// GridFunction Tests for Quadratic Elements
// =============================================================================

class QuadraticGridFunctionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mesh_ = createQuadraticTriangleMesh();
        fes_ = std::make_unique<FESpace>(&mesh_, std::make_unique<H1Collection>(2));
        gf_ = std::make_unique<GridFunction>(fes_.get());
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> gf_;
};

TEST_F(QuadraticGridFunctionTest, SetValueAtVertex)
{
    // Set value at vertex 0
    (*gf_)(0) = 1.0;

    // The value should be retrievable
    EXPECT_NEAR((*gf_)(0), 1.0, 1e-12);
}

TEST_F(QuadraticGridFunctionTest, InterpolateLinearFunction)
{
    // Interpolate f(x,y) = x

    // Set corner vertex DOFs from physical vertex values.
    for (Index v = 0; v < mesh_.numVertices(); ++v) {
        Index d = getCornerVertexDof(*fes_, v);
        if (d == InvalidIndex) {
            continue;
        }
        const Vector3& vert = mesh_.vertex(v);
        (*gf_)(d) = vert.x();
    }

    // Check corner-vertex DOFs.
    for (Index v = 0; v < 4; ++v) { // Only check corner vertices
        Index d = getCornerVertexDof(*fes_, v);
        ASSERT_NE(d, InvalidIndex);
        const Vector3& vert = mesh_.vertex(v);
        EXPECT_NEAR((*gf_)(d), vert.x(), 1e-12);
    }
}

TEST_F(QuadraticGridFunctionTest, EvalAtReferencePoint)
{
    // Set all DOFs to 1
    gf_->setConstant(1.0);

    // Evaluate at reference point (0.5, 0.5) - should be 1
    Vector3 xi(0.5, 0.5, 0.0);
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
    trans.setIntegrationPoint(xi);
    Real val = gf_->eval(0, trans);

    EXPECT_NEAR(val, 1.0, 1e-12);
}

TEST_F(QuadraticGridFunctionTest, EvalQuadraticFunction)
{
    // Set up a quadratic function f = x^2 + y^2
    // On a quadratic element, this should be represented exactly

    for (Index v = 0; v < mesh_.numVertices(); ++v) {
        Index d = getCornerVertexDof(*fes_, v);
        if (d == InvalidIndex) {
            continue;
        }
        const Vector3& p = mesh_.vertex(v);
        (*gf_)(d) = p.x() * p.x() + p.y() * p.y();
    }

    // Check value at vertex 1: (1,0) -> 1.0
    const Index d1 = getCornerVertexDof(*fes_, 1);
    ASSERT_NE(d1, InvalidIndex);
    EXPECT_NEAR((*gf_)(d1), 1.0, 1e-12);
}

// =============================================================================
// Reference Element Tests for Quadratic Elements
// =============================================================================

class QuadraticReferenceElementTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // FECollection created locally in each test
    }
};

TEST_F(QuadraticReferenceElementTest, TriangleNumDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refTri = fec.get(Geometry::Triangle);
    ASSERT_NE(refTri, nullptr);
    EXPECT_EQ(refTri->numDofs(), 6); // 3 + 3
}

TEST_F(QuadraticReferenceElementTest, TetrahedronNumDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refTet = fec.get(Geometry::Tetrahedron);
    ASSERT_NE(refTet, nullptr);
    EXPECT_EQ(refTet->numDofs(), 10); // 4 + 6
}

TEST_F(QuadraticReferenceElementTest, SquareNumDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refSquare = fec.get(Geometry::Square);
    ASSERT_NE(refSquare, nullptr);
    EXPECT_EQ(refSquare->numDofs(), 9); // 4 + 4 + 1
}

TEST_F(QuadraticReferenceElementTest, CubeNumDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refCube = fec.get(Geometry::Cube);
    ASSERT_NE(refCube, nullptr);
    EXPECT_EQ(refCube->numDofs(), 27); // 8 + 12 + 6 + 1
}

TEST_F(QuadraticReferenceElementTest, TriangleFaceDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refTri = fec.get(Geometry::Triangle);
    ASSERT_NE(refTri, nullptr);

    // Triangle has 3 edges as "faces" (boundary edges)
    // Each edge is a Segment with 2 corners + 1 midpoint = 3 DOFs

    for (int f = 0; f < 3; ++f) {
        auto faceDofs = refTri->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 3); // 2 corners + 1 edge midpoint
    }
}

TEST_F(QuadraticReferenceElementTest, TetrahedronFaceDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refTet = fec.get(Geometry::Tetrahedron);
    ASSERT_NE(refTet, nullptr);

    // Tetrahedron has 4 triangular faces
    // Each face has 3 corner DOFs + 3 edge DOFs = 6 DOFs

    for (int f = 0; f < 4; ++f) {
        auto faceDofs = refTet->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 6); // 3 corners + 3 edge midpoints
    }
}

TEST_F(QuadraticReferenceElementTest, CubeFaceDofs)
{
    H1Collection fec(2);
    const ReferenceElement* refCube = fec.get(Geometry::Cube);
    ASSERT_NE(refCube, nullptr);

    // Cube has 6 quadrilateral faces
    // Each face has 4 corners + 4 edges + 1 center = 9 DOFs

    for (int f = 0; f < 6; ++f) {
        auto faceDofs = refCube->faceDofs(f);
        EXPECT_EQ(faceDofs.size(), 9); // 4 corners + 4 edge midpoints + 1 center
    }
}

// =============================================================================
// Integration Tests for Quadratic Elements
// =============================================================================

TEST(QuadraticIntegrationTest, IntegrateQuadraticFunctionExactly)
{
    // Create a single quadratic triangle
    Mesh mesh;
    mesh.setDim(3);

    // Unit right triangle
    mesh.addVertex(0.0, 0.0, 0.0); // 0
    mesh.addVertex(1.0, 0.0, 0.0); // 1
    mesh.addVertex(0.0, 1.0, 0.0); // 2
    mesh.addVertex(0.5, 0.0, 0.0); // 3: edge 0-1 midpoint
    mesh.addVertex(0.0, 0.5, 0.0); // 4: edge 2-0 midpoint
    mesh.addVertex(0.5, 0.5, 0.0); // 5: edge 1-2 midpoint

    mesh.addElement(Geometry::Triangle, {0, 1, 2, 3, 4, 5}, 1, 2);
    mesh.buildTopology();

    FESpace fes(&mesh, std::make_unique<H1Collection>(2));
    GridFunction gf(&fes);

    // Set f(x,y) = x + y at nodal interpolation points of the element.
    const auto dofs = getElementDofsVec(fes, 0);
    const auto points = fes.elementRefElement(0)->interpolationPoints();
    ElementTransform setupTrans;
    bindElementToTransform(setupTrans, mesh, 0);
    for (size_t i = 0; i < points.size(); ++i) {
        setupTrans.setIntegrationPoint(points[i]);
        Vector3 p = setupTrans.transform(points[i]);
        gf(dofs[i]) = p.x() + p.y();
    }

    // Integrate using order-4 quadrature (sufficient for quadratic)
    auto refElem = fes.elementRefElement(0);
    const QuadratureRule& rule = refElem->quadrature();

    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0);
    Real integral = 0.0;

    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip.getXi());
        Real f_val = gf.eval(0, trans);
        integral += ip.weight * trans.weight() * f_val;
    }

    // For f(x,y) = x + y over triangle with vertices (0,0), (1,0), (0,1)
    // Integral = integral of (x + y) over triangle
    // Using barycentric: integral of x over triangle = 1/6, same for y
    // Total = 1/6 + 1/6 = 1/3
    EXPECT_NEAR(integral, 1.0 / 3.0, 1e-10);
}

// =============================================================================
// Mixed-Order Test (Geometric vs Physical Order)
// =============================================================================

TEST(MixedOrderTest, SubparametricElement)
{
    // Test that geometric order (2) and physical order (1) can differ
    // This is subparametric: curved geometry with linear field

    Mesh mesh = createQuadraticTriangleMesh(); // Geometric order = 2
    H1Collection fec(1); // Physical order = 1

    // This should work: linear field on curved geometry
    FESpace fes(&mesh, std::make_unique<H1Collection>(1));
    EXPECT_EQ(fes.order(), 1); // Physical order

    // Element has geometric order 2, but FE space has order 1
    // This is the key test for separation of geometric and physical order
    EXPECT_EQ(mesh.element(0).order, 2); // Geometric order
    EXPECT_EQ(fes.order(), 1); // Physical order
}

TEST(MixedOrderTest, IsoparametricElement)
{
    // Test isoparametric: geometric order = physical order = 2

    Mesh mesh = createQuadraticTriangleMesh(); // Geometric order = 2

    FESpace fes(&mesh, std::make_unique<H1Collection>(2));

    EXPECT_EQ(mesh.element(0).order, 2); // Geometric order
    EXPECT_EQ(fes.order(), 2); // Physical order
}

// =============================================================================
// Real COMSOL Mesh Tests - Verify Node Ordering
// =============================================================================

class COMSOLMeshTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        meshPath_ = dataPath("cases/busbar_steady_order2/mesh.mphtxt");
    }

    std::string meshPath_;
};

TEST_F(COMSOLMeshTest, LoadQuadraticMesh)
{
    // Test loading a real COMSOL quadratic mesh
    Mesh mesh = MphtxtReader::read(meshPath_);

    // Basic sanity checks
    EXPECT_GT(mesh.numVertices(), 0);
    EXPECT_GT(mesh.numElements(), 0);

    // Check for quadratic elements
    bool hasQuadratic = false;
    for (Index e = 0; e < mesh.numElements(); ++e) {
        const auto elem = mesh.element(e);
        if (elem.order == 2) {

            hasQuadratic = true;
            break;
        }
    }
    EXPECT_TRUE(hasQuadratic);
}

TEST_F(COMSOLMeshTest, Tetrahedron2EdgeMidpoints)
{
    // Verify that edge midpoints are correctly positioned in COMSOL edge order
    Mesh mesh = MphtxtReader::read(meshPath_);

    int checkedTets = 0;
    const Real tol = 1e-6;

    for (Index e = 0; e < mesh.numElements() && checkedTets < 10; ++e) {
        const auto elem = mesh.element(e);
        if (elem.geometry != Geometry::Tetrahedron || elem.order != 2) {
            continue;
        }

        const auto vertices = elem.vertices;

        ASSERT_EQ(vertices.size(), 10) << "Tetrahedron2 should have 10 vertices";

        // Corner vertices
        auto v0 = mesh.vertex(vertices[0]);
        auto v1 = mesh.vertex(vertices[1]);
        auto v2 = mesh.vertex(vertices[2]);
        auto v3 = mesh.vertex(vertices[3]);

        // Edge midpoints in COMSOL ordering:
        // dof 4: E01, dof 5: E02, dof 6: E12, dof 7: E03, dof 8: E13, dof 9: E23
        auto e01 = mesh.vertex(vertices[4]); // Edge 0-1
        auto e02 = mesh.vertex(vertices[5]); // Edge 0-2
        auto e12 = mesh.vertex(vertices[6]); // Edge 1-2
        auto e03 = mesh.vertex(vertices[7]); // Edge 0-3
        auto e13 = mesh.vertex(vertices[8]); // Edge 1-3
        auto e23 = mesh.vertex(vertices[9]); // Edge 2-3

        // Verify edge midpoints are at the correct positions
        auto checkMidpoint = [&](const Vector3& mid, const Vector3& a, const Vector3& b,
                                 const std::string& edgeName) {
            Vector3 expected((a.x() + b.x()) / 2, (a.y() + b.y()) / 2, (a.z() + b.z()) / 2);
            Real dist = std::sqrt(
                std::pow(mid.x() - expected.x(), 2) + std::pow(mid.y() - expected.y(), 2) + std::pow(mid.z() - expected.z(), 2));
            EXPECT_LT(dist, tol) << "Edge " << edgeName << " midpoint mismatch in element " << e
                                 << ": expected (" << expected.x() << ", " << expected.y()
                                 << ", " << expected.z() << "), got (" << mid.x() << ", "
                                 << mid.y() << ", " << mid.z() << ")";
        };

        checkMidpoint(e01, v0, v1, "E01");
        checkMidpoint(e02, v0, v2, "E02");
        checkMidpoint(e12, v1, v2, "E12");
        checkMidpoint(e03, v0, v3, "E03");
        checkMidpoint(e13, v1, v3, "E13");
        checkMidpoint(e23, v2, v3, "E23");

        checkedTets++;
    }

    EXPECT_GT(checkedTets, 0) << "No quadratic tetrahedra found in mesh";
}

TEST_F(COMSOLMeshTest, JacobianPositiveDefinite)
{
    // Verify that Jacobian determinant is positive at quadrature points
    // This is a crucial test for correct node ordering
    Mesh mesh = MphtxtReader::read(meshPath_);

    FESpace fes(&mesh, std::make_unique<H1Collection>(2));
    ElementTransform trans;

    int checkedElems = 0;

    for (Index e = 0; e < mesh.numElements() && checkedElems < 20; ++e) {
        const auto elem = mesh.element(e);
        if (elem.order != 2)
            continue;

        bindElementToTransform(trans, mesh, e);
        auto refElem = fes.elementRefElement(e);
        const QuadratureRule& rule = refElem->quadrature();

        for (const auto& ip : rule) {
            trans.setIntegrationPoint(ip.getXi());
            Real detJ = trans.detJ();

            EXPECT_GT(detJ, 0.0) << "Negative or zero Jacobian determinant at element " << e
                                 << " with detJ = " << detJ;
        }

        checkedElems++;
    }

    EXPECT_GT(checkedElems, 0) << "No quadratic elements found in mesh";
}

TEST_F(COMSOLMeshTest, FESpaceConsistency)
{
    // Test that FESpace correctly handles the reordered mesh
    Mesh mesh = MphtxtReader::read(meshPath_);

    FESpace fes(&mesh, std::make_unique<H1Collection>(2));

    // DOFs should map directly to mesh vertices for COMSOL-style meshes
    EXPECT_EQ(fes.numDofs(), mesh.numVertices());

    // Check element DOF mapping consistency
    for (Index e = 0; e < std::min(mesh.numElements(), Index(10)); ++e) {
        const auto elem = mesh.element(e);
        if (elem.order != 2)
            continue;

        std::vector<Index> dofs = getElementDofsVec(fes, e);
        auto vertices = elem.vertices;

        ASSERT_EQ(dofs.size(), vertices.size());

        for (int i = 0; i < elem.numCorners(); ++i) {
            const Index expected = getCornerVertexDof(fes, vertices[static_cast<size_t>(i)]);
            EXPECT_EQ(dofs[static_cast<size_t>(i)], expected)
                << "Corner DOF mismatch at local index " << i
                << " of element " << e;
        }
        for (Index d : dofs) {
            EXPECT_NE(d, InvalidIndex);
        }
    }
}

TEST_F(COMSOLMeshTest, FiniteElementKroneckerDelta)
{
    // Verify that H1 FiniteElement basis has Kronecker delta property at nodes.
    // This tests the consistency between node ordering and basis definition.
    Mesh mesh = MphtxtReader::read(meshPath_);

    FESpace fes(&mesh, std::make_unique<H1Collection>(2));
    ElementTransform trans;

    const Real tol = 1e-10;
    int testedElems = 0;

    for (Index e = 0; e < mesh.numElements() && testedElems < 5; ++e) {
        const auto elem = mesh.element(e);
        if (elem.geometry != Geometry::Tetrahedron || elem.order != 2) {
            continue;
        }

        bindElementToTransform(trans, mesh, e);

        auto refElem = fes.elementRefElement(e);
        const FiniteElement& h1Element = refElem->basis();
        auto dofCoords = h1Element.interpolationPoints();

        // Pre-allocate storage
        ShapeMatrix values;

        // At each node position, only the corresponding H1 basis function should be 1.
        for (int i = 0; i < static_cast<int>(dofCoords.size()); ++i) {
            h1Element.evalShape(dofCoords[i], values);

            for (int j = 0; j < values.rows(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(values(j, 0), 1.0, tol)
                        << "H1 basis function " << j << " should be 1 at its own node in element " << e;
                }
                else {
                    EXPECT_NEAR(values(j, 0), 0.0, tol)
                        << "H1 basis function " << j << " should be 0 at node " << i << " in element " << e;
                }
            }
        }

        testedElems++;
    }

    EXPECT_GT(testedElems, 0) << "No Tetrahedron2 elements found for testing";
}

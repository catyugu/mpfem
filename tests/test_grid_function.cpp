#include "assembly/element_binding.hpp"
#include "core/logger.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include <gtest/gtest.h>


using namespace mpfem;

class GridFunctionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Logger::setLevel(LogLevel::Debug);
    }

    // Create a simple 1-tetrahedron mesh for testing
    Mesh createSimpleTetMesh()
    {
        Mesh mesh;
        mesh.setDim(3);

        // Add 4 vertices forming a tetrahedron
        mesh.addVertex(0.0, 0.0, 0.0); // Vertex 0
        mesh.addVertex(1.0, 0.0, 0.0); // Vertex 1
        mesh.addVertex(0.0, 1.0, 0.0); // Vertex 2
        mesh.addVertex(0.0, 0.0, 1.0); // Vertex 3

        // Add 1 tetrahedron
        mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1);

        mesh.buildTopology();

        return mesh;
    }
};

TEST_F(GridFunctionTest, Construction)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 1);

    GridFunction gf(&fes);

    EXPECT_EQ(gf.numDofs(), 4); // 4 vertices for 1 tet
    EXPECT_EQ(gf.fes(), &fes);
}

TEST_F(GridFunctionTest, Initialization)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 1);

    GridFunction gf(&fes, 3.14);

    for (Index i = 0; i < gf.numDofs(); ++i) {
        EXPECT_DOUBLE_EQ(gf(i), 3.14);
    }
}

TEST_F(GridFunctionTest, SetValues)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 1);

    GridFunction gf(&fes);
    gf.setConstant(1.0);

    EXPECT_DOUBLE_EQ(gf.l2Norm(), 2.0); // sqrt(4 * 1^2) = 2
    EXPECT_DOUBLE_EQ(gf.maxNorm(), 1.0);
    EXPECT_DOUBLE_EQ(gf.minValue(), 1.0);
    EXPECT_DOUBLE_EQ(gf.maxValue(), 1.0);
}

TEST_F(GridFunctionTest, InterpolationLinear)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 1);

    GridFunction gf(&fes);

    // Set values at vertices: [0, 1, 2, 3]
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 2.0, 3.0;
    gf.setValues(values);

    // Evaluate at barycentric coordinates (0.25, 0.25, 0.25, 0.25)
    // i.e., at the center of the tetrahedron
    // Expected value: 0.25*0 + 0.25*1 + 0.25*2 + 0.25*3 = 1.5
    Real xi[] = {0.25, 0.25, 0.25}; // Barycentric coords (xi1, xi2, xi3)
    Vector3 xi_vec(xi[0], xi[1], xi[2]);
    ElementTransform trans;
    bindElementToTransform(trans, mesh, 0);
    trans.setIntegrationPoint(xi_vec);
    Real value = gf.eval(0, trans);

    // For reference tet, the barycentric coords are:
    // lambda0 = 1 - xi1 - xi2 - xi3
    // lambda1 = xi1
    // lambda2 = xi2
    // lambda3 = xi3
    // Value = lambda0*0 + lambda1*1 + lambda2*2 + lambda3*3
    // At (0.25, 0.25, 0.25): value = 0.25*0 + 0.25*1 + 0.25*2 + 0.25*3 = 1.5
    EXPECT_NEAR(value, 1.5, 1e-10);
}

TEST_F(GridFunctionTest, ElementValues)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 1);

    GridFunction gf(&fes);
    gf.setConstant(1.0);

    // Get element values
    std::array<Real, MaxDofsPerElement> elemVals {};
    gf.getElementValues(0, std::span<Real>(elemVals.data(), 4));

    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(elemVals[i], 1.0);
    }
}

TEST_F(GridFunctionTest, VectorField)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(1), 3); // vdim = 3 for vector field

    GridFunction gf(&fes);
    EXPECT_EQ(gf.vdim(), 3);
    EXPECT_EQ(gf.numDofs(), 12); // 4 vertices * 3 components
}

TEST_F(GridFunctionTest, QuadraticElement)
{
    Mesh mesh = createSimpleTetMesh();
    FESpace fes(&mesh, std::make_unique<FECollection>(2), 1);

    GridFunction gf(&fes);

    // Topology-based quadratic tetrahedron DOFs: 4 vertex + 6 edge = 10
    EXPECT_EQ(gf.numDofs(), 10);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

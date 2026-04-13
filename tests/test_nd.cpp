#include "core/geometry.hpp"
#include "core/types.hpp"
#include "fe/nd.hpp"
#include "fe/shape_evaluator.hpp"
#include "field/fe_space.hpp"
#include "mesh/mesh.hpp"

#include <array>
#include <gtest/gtest.h>

using namespace mpfem;

namespace {

    std::array<Vector3, 4> refTetVertices()
    {
        return {
            Vector3(0.0, 0.0, 0.0),
            Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 1.0, 0.0),
            Vector3(0.0, 0.0, 1.0)};
    }

    Real edgeLineIntegral(const NDFiniteElement& nd, int basisIdx, int edgeIdx)
    {
        const auto verts = refTetVertices();
        const auto [ea, eb] = geom::edgeVertices(Geometry::Tetrahedron, edgeIdx);
        const Vector3 a = verts[ea];
        const Vector3 b = verts[eb];
        const Vector3 tangent = b - a;

        // 3-point Gauss rule on [0, 1].
        constexpr Real tPts[3] = {
            0.5 * (1.0 - 0.7745966692414834),
            0.5,
            0.5 * (1.0 + 0.7745966692414834)};
        constexpr Real tWts[3] = {
            5.0 / 18.0,
            8.0 / 18.0,
            5.0 / 18.0};

        Real value = 0.0;
        ShapeMatrix shape;
        for (int q = 0; q < 3; ++q) {
            const Vector3 xq = a + tPts[q] * tangent;
            nd.evalShape(xq, shape);
            const Vector3 N(shape(basisIdx, 0), shape(basisIdx, 1), shape(basisIdx, 2));
            value += tWts[q] * N.dot(tangent);
        }
        return value;
    }

} // namespace

TEST(NDFiniteElementTest, KroneckerEdgeIntegralOnReferenceTet)
{
    NDFiniteElement nd(Geometry::Tetrahedron, 1);

    Matrix kr = Matrix::Zero(6, 6);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            kr(i, j) = edgeLineIntegral(nd, i, j);
        }
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const Real target = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(kr(i, j), target, 1e-10) << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FESpaceNDTest, EdgeOrientationFlipsOnReversedLocalEdge)
{
    Mesh mesh;
    mesh.setDim(3);

    mesh.addNode(0.0, 0.0, 0.0); // 0
    mesh.addNode(1.0, 0.0, 0.0); // 1
    mesh.addNode(0.0, 1.0, 0.0); // 2
    mesh.addNode(0.0, 0.0, 1.0); // 3
    mesh.addNode(1.0, 1.0, 1.0); // 4

    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3});
    mesh.addElement(Geometry::Tetrahedron, {1, 0, 2, 4});
    mesh.buildTopology();

    FESpace fes(&mesh, std::make_unique<NDCollection>(1));

    const auto o0 = fes.getElementOrientations(0);
    const auto o1 = fes.getElementOrientations(1);

    ASSERT_EQ(o0.size(), 6u);
    ASSERT_EQ(o1.size(), 6u);

    EXPECT_EQ(o0[0], 1);
    EXPECT_EQ(o1[0], -1);
}

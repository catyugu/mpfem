#include "h1.hpp"
#include "core/exception.hpp"
#include <algorithm>

namespace mpfem {

    namespace {

        // =========================================================================
        // 拓扑映射与数学抽象 (Topology & Mathematical Abstractions)
        // =========================================================================

        inline bool isTensorProduct(Geometry g)
        {
            return g == Geometry::Segment || g == Geometry::Square || g == Geometry::Cube;
        }

        // --- 拓扑映射表：将局部 DoF 索引映射到 1D 基底/重心坐标 的组合 ---
        namespace topo {
            // 张量积节点映射 (基于 1D 节点的排列组合: 0=Left(0), 1=Center(0.5), 2=Right(1))
            constexpr int Seg_O1[2][1] = {{0}, {1}};
            constexpr int Seg_O2[3][1] = {{0}, {1}, {2}}; // V0, Center, V1

            constexpr int Quad_O1[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            constexpr int Quad_O2[9][2] = {
                {0, 0}, {2, 0}, {2, 2}, {0, 2}, // Vertices
                {1, 0}, {2, 1}, {1, 2}, {0, 1}, // Edges
                {1, 1} // Face/Center
            };

            constexpr int Hex_O1[8][3] = {
                {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            constexpr int Hex_O2[27][3] = {
                {0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}, {0, 0, 2}, {2, 0, 2}, {2, 2, 2}, {0, 2, 2}, // Vertices
                {1, 0, 0}, {2, 1, 0}, {1, 2, 0}, {0, 1, 0}, // Edges (z=-1)
                {0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}, // Edges (z=0)
                {1, 0, 2}, {2, 1, 2}, {1, 2, 2}, {0, 1, 2}, // Edges (z=1)
                {1, 1, 0}, {1, 0, 1}, {2, 1, 1}, {1, 2, 1}, {0, 1, 1}, {1, 1, 2}, // Faces
                {1, 1, 1} // Center
            };

            // 单纯形节点映射 (重心坐标对索引。若两索引相同为顶点，不同则为边的中点)
            constexpr int Tri_O1[3][2] = {{0, 0}, {1, 1}, {2, 2}};
            constexpr int Tri_O2[6][2] = {
                {0, 0}, {1, 1}, {2, 2}, // Vertices
                {1, 0}, {2, 0}, {1, 2} // Edges
            };

            constexpr int Tet_O1[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
            constexpr int Tet_O2[10][2] = {
                {0, 0}, {1, 1}, {2, 2}, {3, 3}, // Vertices
                {1, 0}, {2, 0}, {1, 2}, {3, 0}, {1, 3}, {2, 3} // Edges
            };
        }

        // --- 1. 张量积基底引擎：1D 拉格朗日多项式求值 ---
        struct Lagrange1D {
            Real N[3], dN[3];
            Lagrange1D(int order, Real x)
            {
                if (order == 1) {
                    N[0] = 1.0 - x; // Node at x=0: =1 at x=0, =0 at x=1
                    dN[0] = -1.0;
                    N[1] = x; // Node at x=1: =0 at x=0, =1 at x=1
                    dN[1] = 1.0;
                }
                else {
                    N[0] = (1.0 - x) * (1.0 - 2.0 * x); // Node at x=0
                    dN[0] = 4.0 * x - 3.0;
                    N[1] = 4.0 * x * (1.0 - x); // Node at x=0.5
                    dN[1] = 4.0 - 8.0 * x;
                    N[2] = x * (2.0 * x - 1.0); // Node at x=1
                    dN[2] = 4.0 * x - 1.0;
                }
            }
        };

        // --- 2. 单纯形基底引擎：重心坐标求值 ---
        struct Barycentric {
            Real L[4];
            Real dL[4][3]; // dL[节点][空间维度]

            Barycentric(Geometry geom, const Vector3& xi)
            {
                const Real x = xi.x(), y = xi.y(), z = xi.z();
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 3; ++j)
                        dL[i][j] = 0.0;

                if (geom == Geometry::Triangle) {
                    L[0] = 1.0 - x - y;
                    dL[0][0] = -1.0;
                    dL[0][1] = -1.0;
                    L[1] = x;
                    dL[1][0] = 1.0;
                    dL[1][1] = 0.0;
                    L[2] = y;
                    dL[2][0] = 0.0;
                    dL[2][1] = 1.0;
                }
                else if (geom == Geometry::Tetrahedron) {
                    L[0] = 1.0 - x - y - z;
                    dL[0][0] = -1.0;
                    dL[0][1] = -1.0;
                    dL[0][2] = -1.0;
                    L[1] = x;
                    dL[1][0] = 1.0;
                    dL[1][1] = 0.0;
                    dL[1][2] = 0.0;
                    L[2] = y;
                    dL[2][0] = 0.0;
                    dL[2][1] = 1.0;
                    dL[2][2] = 0.0;
                    L[3] = z;
                    dL[3][0] = 0.0;
                    dL[3][1] = 0.0;
                    dL[3][2] = 1.0;
                }
            }
        };

        // =========================================================================
        // 布局与尺寸工具
        // =========================================================================

        inline DofLayout h1DofLayout(Geometry g, int order)
        {
            if (g == Geometry::Square)
                return DofLayout {1, std::max(0, order - 1), order > 1 ? 1 : 0, 0};
            if (g == Geometry::Cube)
                return DofLayout {1, std::max(0, order - 1), order > 1 ? 1 : 0, order > 1 ? 1 : 0};
            return DofLayout {1, std::max(0, order - 1), 0, 0};
        }

        inline int h1NumDofs(Geometry g, int order)
        {
            switch (g) {
            case Geometry::Segment:
                return order + 1;
            case Geometry::Triangle:
                return (order + 1) * (order + 2) / 2;
            case Geometry::Square:
                return (order + 1) * (order + 1);
            case Geometry::Tetrahedron:
                return (order + 1) * (order + 2) * (order + 3) / 6;
            case Geometry::Cube:
                return (order + 1) * (order + 1) * (order + 1);
            default:
                MPFEM_THROW(Exception, "H1FiniteElement: unsupported geometry");
            }
        }

    } // namespace

    // =========================================================================
    // H1FiniteElement 核心实现
    // =========================================================================

    H1FiniteElement::H1FiniteElement(Geometry geom, int order, int vdim)
        : geom_(geom), order_(order), numDofs_(h1NumDofs(geom, order)), vdim_(vdim)
    {
    }

    int H1FiniteElement::numDofs() const { return numDofs_; }
    DofLayout H1FiniteElement::dofLayout() const { return h1DofLayout(geom_, order_); }

    std::vector<Vector3> H1FiniteElement::interpolationPoints() const
    {
        std::vector<Vector3> pts(numDofs_);
        if (isTensorProduct(geom_)) {
            const Real v1[2] = {0.0, 1.0}, v2[3] = {0.0, 0.5, 1.0};
            const Real* v = (order_ == 1) ? v1 : v2;
            for (int i = 0; i < numDofs_; ++i) {
                if (geom_ == Geometry::Segment) {
                    pts[i] = Vector3(v[(order_ == 1) ? topo::Seg_O1[i][0] : topo::Seg_O2[i][0]], 0, 0);
                }
                else if (geom_ == Geometry::Square) {
                    int ix = (order_ == 1) ? topo::Quad_O1[i][0] : topo::Quad_O2[i][0];
                    int iy = (order_ == 1) ? topo::Quad_O1[i][1] : topo::Quad_O2[i][1];
                    pts[i] = Vector3(v[ix], v[iy], 0);
                }
                else if (geom_ == Geometry::Cube) {
                    int ix = (order_ == 1) ? topo::Hex_O1[i][0] : topo::Hex_O2[i][0];
                    int iy = (order_ == 1) ? topo::Hex_O1[i][1] : topo::Hex_O2[i][1];
                    int iz = (order_ == 1) ? topo::Hex_O1[i][2] : topo::Hex_O2[i][2];
                    pts[i] = Vector3(v[ix], v[iy], v[iz]);
                }
            }
        }
        else {
            const Vector3 vTri[3] = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};
            const Vector3 vTet[4] = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};
            const Vector3* v = (geom_ == Geometry::Triangle) ? vTri : vTet;
            for (int i = 0; i < numDofs_; ++i) {
                if (order_ == 1) {
                    pts[i] = v[(geom_ == Geometry::Triangle) ? topo::Tri_O1[i][0] : topo::Tet_O1[i][0]];
                }
                else {
                    int n1 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][0] : topo::Tet_O2[i][0];
                    int n2 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][1] : topo::Tet_O2[i][1];
                    pts[i] = (n1 == n2) ? v[n1] : 0.5 * (v[n1] + v[n2]);
                }
            }
        }
        return pts;
    }

    void H1FiniteElement::evalShape(const Vector3& xi, ShapeMatrix& shape) const
    {
        if (shape.rows() != numDofs_ || shape.cols() != 1)
            shape.resize(numDofs_, 1);
        Real* s = shape.data();

        if (isTensorProduct(geom_)) {
            Lagrange1D bx(order_, xi.x()), by(order_, xi.y()), bz(order_, xi.z());
            if (geom_ == Geometry::Segment) {
                for (int i = 0; i < numDofs_; ++i)
                    s[i] = bx.N[(order_ == 1) ? topo::Seg_O1[i][0] : topo::Seg_O2[i][0]];
            }
            else if (geom_ == Geometry::Square) {
                for (int i = 0; i < numDofs_; ++i) {
                    int ix = (order_ == 1) ? topo::Quad_O1[i][0] : topo::Quad_O2[i][0];
                    int iy = (order_ == 1) ? topo::Quad_O1[i][1] : topo::Quad_O2[i][1];
                    s[i] = bx.N[ix] * by.N[iy];
                }
            }
            else if (geom_ == Geometry::Cube) {
                for (int i = 0; i < numDofs_; ++i) {
                    int ix = (order_ == 1) ? topo::Hex_O1[i][0] : topo::Hex_O2[i][0];
                    int iy = (order_ == 1) ? topo::Hex_O1[i][1] : topo::Hex_O2[i][1];
                    int iz = (order_ == 1) ? topo::Hex_O1[i][2] : topo::Hex_O2[i][2];
                    s[i] = bx.N[ix] * by.N[iy] * bz.N[iz];
                }
            }
        }
        else {
            Barycentric b(geom_, xi);
            for (int i = 0; i < numDofs_; ++i) {
                if (order_ == 1) {
                    s[i] = b.L[(geom_ == Geometry::Triangle) ? topo::Tri_O1[i][0] : topo::Tet_O1[i][0]];
                }
                else {
                    int n1 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][0] : topo::Tet_O2[i][0];
                    int n2 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][1] : topo::Tet_O2[i][1];
                    s[i] = (n1 == n2) ? b.L[n1] * (2.0 * b.L[n1] - 1.0) : 4.0 * b.L[n1] * b.L[n2];
                }
            }
        }
    }

    void H1FiniteElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        int d = dim();
        if (derivatives.rows() != numDofs_ || derivatives.cols() != d)
            derivatives.resize(numDofs_, d);

        if (isTensorProduct(geom_)) {
            Lagrange1D bx(order_, xi.x()), by(order_, xi.y()), bz(order_, xi.z());
            if (geom_ == Geometry::Segment) {
                for (int i = 0; i < numDofs_; ++i)
                    derivatives(i, 0) = bx.dN[(order_ == 1) ? topo::Seg_O1[i][0] : topo::Seg_O2[i][0]];
            }
            else if (geom_ == Geometry::Square) {
                for (int i = 0; i < numDofs_; ++i) {
                    int ix = (order_ == 1) ? topo::Quad_O1[i][0] : topo::Quad_O2[i][0];
                    int iy = (order_ == 1) ? topo::Quad_O1[i][1] : topo::Quad_O2[i][1];
                    derivatives(i, 0) = bx.dN[ix] * by.N[iy];
                    derivatives(i, 1) = bx.N[ix] * by.dN[iy];
                }
            }
            else if (geom_ == Geometry::Cube) {
                for (int i = 0; i < numDofs_; ++i) {
                    int ix = (order_ == 1) ? topo::Hex_O1[i][0] : topo::Hex_O2[i][0];
                    int iy = (order_ == 1) ? topo::Hex_O1[i][1] : topo::Hex_O2[i][1];
                    int iz = (order_ == 1) ? topo::Hex_O1[i][2] : topo::Hex_O2[i][2];
                    derivatives(i, 0) = bx.dN[ix] * by.N[iy] * bz.N[iz];
                    derivatives(i, 1) = bx.N[ix] * by.dN[iy] * bz.N[iz];
                    derivatives(i, 2) = bx.N[ix] * by.N[iy] * bz.dN[iz];
                }
            }
        }
        else {
            Barycentric b(geom_, xi);
            for (int i = 0; i < numDofs_; ++i) {
                if (order_ == 1) {
                    int n = (geom_ == Geometry::Triangle) ? topo::Tri_O1[i][0] : topo::Tet_O1[i][0];
                    for (int k = 0; k < d; ++k)
                        derivatives(i, k) = b.dL[n][k];
                }
                else {
                    int n1 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][0] : topo::Tet_O2[i][0];
                    int n2 = (geom_ == Geometry::Triangle) ? topo::Tri_O2[i][1] : topo::Tet_O2[i][1];

                    if (n1 == n2) {
                        Real coeff = 4.0 * b.L[n1] - 1.0;
                        for (int k = 0; k < d; ++k)
                            derivatives(i, k) = coeff * b.dL[n1][k];
                    }
                    else {
                        for (int k = 0; k < d; ++k)
                            derivatives(i, k) = 4.0 * (b.L[n2] * b.dL[n1][k] + b.L[n1] * b.dL[n2][k]);
                    }
                }
            }
        }
    }

    std::vector<int> H1FiniteElement::vertexDofs(int vertexIdx) const
    {
        if (vertexIdx < 0 || vertexIdx >= geom::numVertices(geom_)) {
            return {};
        }
        return {vertexIdx};
    }

    std::vector<int> H1FiniteElement::edgeDofs(int edgeIdx) const
    {
        std::vector<int> dofs;
        const DofLayout layout = h1DofLayout(geom_, order_);

        if (geom_ == Geometry::Point) {
            return dofs;
        }

        if (geom_ == Geometry::Segment) {
            if (edgeIdx != 0) {
                return dofs;
            }
            dofs = {0, 1};
            if (order_ > 1) {
                for (int j = 0; j < layout.numEdgeDofs; ++j) {
                    dofs.push_back(geom::numVertices(geom_) + j);
                }
            }
            return dofs;
        }

        if (edgeIdx < 0 || edgeIdx >= geom::numEdges(geom_)) {
            return dofs;
        }

        auto [v0, v1] = geom::edgeVertices(geom_, edgeIdx);
        dofs.push_back(v0);
        dofs.push_back(v1);

        if (order_ <= 1 || layout.numEdgeDofs <= 0) {
            return dofs;
        }

        const int edgeBase = geom::numVertices(geom_);
        const int base = edgeBase + edgeIdx * layout.numEdgeDofs;
        for (int j = 0; j < layout.numEdgeDofs; ++j) {
            dofs.push_back(base + j);
        }

        return dofs;
    }

    std::vector<int> H1FiniteElement::faceDofs(int faceIdx) const
    {
        std::vector<int> dofs;
        const DofLayout layout = h1DofLayout(geom_, order_);

        if (geom_ == Geometry::Segment) {
            return dofs;
        }

        dofs = geom::faceVertices(geom_, faceIdx);
        if (dofs.empty() || order_ <= 1)
            return dofs;

        const std::vector<int> faceEdges = geom::faceEdges(geom_, faceIdx);
        const int edgeDofs = layout.numEdgeDofs;
        const int edgeBase = geom::numVertices(geom_);

        for (int edgeIdx : faceEdges) {
            const int base = edgeBase + edgeIdx * edgeDofs;
            for (int j = 0; j < edgeDofs; ++j)
                dofs.push_back(base + j);
        }

        if (geom_ == Geometry::Cube && layout.numFaceDofs > 0) {
            const int faceBase = edgeBase + geom::numEdges(geom_) * edgeDofs;
            for (int j = 0; j < layout.numFaceDofs; ++j) {
                dofs.push_back(faceBase + faceIdx * layout.numFaceDofs + j);
            }
        }
        if (geom_ == Geometry::Square && layout.numFaceDofs > 0 && faceIdx == 0) {
            const int faceBase = edgeBase + geom::numEdges(geom_) * edgeDofs;
            for (int j = 0; j < layout.numFaceDofs; ++j) {
                dofs.push_back(faceBase + j);
            }
        }

        return dofs;
    }

    std::vector<int> H1FiniteElement::cellDofs(int cellIdx) const
    {
        if (cellIdx != 0 || geom::dim(geom_) != 3) {
            return {};
        }

        const DofLayout layout = h1DofLayout(geom_, order_);
        if (layout.numVolumeDofs <= 0) {
            return {};
        }

        const int edgeBase = geom::numVertices(geom_);
        const int faceBase = edgeBase + geom::numEdges(geom_) * layout.numEdgeDofs;
        const int cellBase = faceBase + geom::numFaces(geom_) * layout.numFaceDofs;

        std::vector<int> dofs;
        dofs.reserve(static_cast<size_t>(layout.numVolumeDofs));
        for (int j = 0; j < layout.numVolumeDofs; ++j) {
            dofs.push_back(cellBase + j);
        }
        return dofs;
    }

} // namespace mpfem
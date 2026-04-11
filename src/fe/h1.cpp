#include "h1.hpp"

#include "core/exception.hpp"
#include "fe/geometry_mapping.hpp"

namespace mpfem {

    namespace {

        template <typename FEType>
        std::vector<int> buildH1FaceDofs(const FEType& fe, int faceIdx)
        {
            std::vector<int> dofs;
            const Geometry g = fe.geometry();
            const DofLayout layout = fe.dofLayout();

            if (g == Geometry::Segment) {
                if (faceIdx == 0) {
                    return {0};
                }
                if (faceIdx == 1) {
                    return {1};
                }
                return dofs;
            }

            const std::vector<int> faceVerts = geom::faceVertices(g, faceIdx);
            if (faceVerts.empty()) {
                return dofs;
            }

            dofs = faceVerts;
            if (fe.order() <= 1) {
                return dofs;
            }

            const std::vector<int> faceEdges = geom::faceEdges(g, faceIdx);
            const int edgeDofs = layout.numEdgeDofs;
            const int edgeBase = geom::numCorners(g);

            for (int edgeIdx : faceEdges) {
                const int base = edgeBase + edgeIdx * edgeDofs;
                for (int j = 0; j < edgeDofs; ++j) {
                    dofs.push_back(base + j);
                }
            }

            if (g == Geometry::Cube) {
                const int faceBase = edgeBase + geom::numEdges(g) * edgeDofs;
                for (int j = 0; j < layout.numFaceDofs; ++j) {
                    dofs.push_back(faceBase + faceIdx * layout.numFaceDofs + j);
                }
            }

            return dofs;
        }

    } // namespace

    // =============================================================================
    // H1SegmentShape
    // =============================================================================

    H1SegmentShape::H1SegmentShape(int order)
        : order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1SegmentShape: only order 1 and 2 supported");
        }
    }

    void H1SegmentShape::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geometry(), order_, xi, shape);
    }

    void H1SegmentShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geometry(), order_, xi, derivatives);
    }

    std::vector<int> H1SegmentShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    std::vector<Vector3> H1SegmentShape::interpolationPoints() const
    {
        if (order_ == 1) {
            return {Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
        }
        return {Vector3(-1.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
    }

    // =============================================================================
    // H1TriangleShape
    // =============================================================================

    H1TriangleShape::H1TriangleShape(int order)
        : order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1TriangleShape: only order 1 and 2 supported");
        }
    }

    int H1TriangleShape::numDofs() const
    {
        return (order_ + 1) * (order_ + 2) / 2;
    }

    void H1TriangleShape::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geometry(), order_, xi, shape);
    }

    void H1TriangleShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geometry(), order_, xi, derivatives);
    }

    std::vector<int> H1TriangleShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    std::vector<Vector3> H1TriangleShape::interpolationPoints() const
    {
        if (order_ == 1) {
            return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)};
        }
        return {
            Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0),
            Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0)};
    }

    // =============================================================================
    // H1SquareShape
    // =============================================================================

    H1SquareShape::H1SquareShape(int order)
        : order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1SquareShape: only order 1 and 2 supported");
        }
    }

    void H1SquareShape::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geometry(), order_, xi, shape);
    }

    void H1SquareShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geometry(), order_, xi, derivatives);
    }

    std::vector<int> H1SquareShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    std::vector<Vector3> H1SquareShape::interpolationPoints() const
    {
        if (order_ == 1) {
            return {Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0)};
        }
        return {
            Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0),
            Vector3(0.0, -1.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(-1.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0)};
    }

    // =============================================================================
    // H1TetrahedronShape
    // =============================================================================

    H1TetrahedronShape::H1TetrahedronShape(int order)
        : order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1TetrahedronShape: only order 1 and 2 supported");
        }
    }

    int H1TetrahedronShape::numDofs() const
    {
        return (order_ + 1) * (order_ + 2) * (order_ + 3) / 6;
    }

    void H1TetrahedronShape::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geometry(), order_, xi, shape);
    }

    void H1TetrahedronShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geometry(), order_, xi, derivatives);
    }

    std::vector<int> H1TetrahedronShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    std::vector<Vector3> H1TetrahedronShape::interpolationPoints() const
    {
        if (order_ == 1) {
            return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)};
        }
        return {
            Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0),
            Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0),
            Vector3(0.0, 0.0, 0.5), Vector3(0.5, 0.0, 0.5), Vector3(0.0, 0.5, 0.5)};
    }

    // =============================================================================
    // H1CubeShape
    // =============================================================================

    H1CubeShape::H1CubeShape(int order)
        : order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1CubeShape: only order 1 and 2 supported");
        }
    }

    void H1CubeShape::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geometry(), order_, xi, shape);
    }

    void H1CubeShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geometry(), order_, xi, derivatives);
    }

    std::vector<int> H1CubeShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    std::vector<Vector3> H1CubeShape::interpolationPoints() const
    {
        if (order_ == 1) {
            return {
                Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), Vector3(1.0, 1.0, -1.0), Vector3(-1.0, 1.0, -1.0),
                Vector3(-1.0, -1.0, 1.0), Vector3(1.0, -1.0, 1.0), Vector3(1.0, 1.0, 1.0), Vector3(-1.0, 1.0, 1.0)};
        }

        return {
            Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), Vector3(1.0, 1.0, -1.0), Vector3(-1.0, 1.0, -1.0),
            Vector3(-1.0, -1.0, 1.0), Vector3(1.0, -1.0, 1.0), Vector3(1.0, 1.0, 1.0), Vector3(-1.0, 1.0, 1.0),
            Vector3(0.0, -1.0, -1.0), Vector3(1.0, 0.0, -1.0), Vector3(0.0, 1.0, -1.0), Vector3(-1.0, 0.0, -1.0),
            Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0),
            Vector3(0.0, -1.0, 1.0), Vector3(1.0, 0.0, 1.0), Vector3(0.0, 1.0, 1.0), Vector3(-1.0, 0.0, 1.0),
            Vector3(0.0, 0.0, -1.0), Vector3(0.0, -1.0, 0.0), Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 1.0, 0.0), Vector3(-1.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0),
            Vector3(0.0, 0.0, 0.0)};
    }

} // namespace mpfem

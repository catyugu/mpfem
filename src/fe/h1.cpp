#include "h1.hpp"

#include "core/exception.hpp"
#include "fe/geometry_mapping.hpp"
#include <algorithm>

namespace mpfem {

    namespace {

        DofLayout h1DofLayout(Geometry g, int order)
        {
            if (g == Geometry::Square) {
                return DofLayout {1, std::max(0, order - 1), order > 1 ? 1 : 0, 0};
            }
            if (g == Geometry::Cube) {
                return DofLayout {1, std::max(0, order - 1), order > 1 ? 1 : 0, order > 1 ? 1 : 0};
            }
            return DofLayout {1, std::max(0, order - 1), 0, 0};
        }

        int h1NumDofs(Geometry g, int order)
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
                MPFEM_THROW(Exception, "H1FiniteElement unsupported geometry");
            }
        }

        std::vector<int> buildH1FaceDofs(Geometry g, int order, int faceIdx)
        {
            std::vector<int> dofs;
            const DofLayout layout = h1DofLayout(g, order);

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
            if (order <= 1) {
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

            if (g == Geometry::Cube && layout.numFaceDofs > 0) {
                const int faceBase = edgeBase + geom::numEdges(g) * edgeDofs;
                for (int j = 0; j < layout.numFaceDofs; ++j) {
                    dofs.push_back(faceBase + faceIdx * layout.numFaceDofs + j);
                }
            }

            return dofs;
        }

        std::vector<Vector3> buildInterpolationPoints(Geometry g, int order)
        {
            switch (g) {
            case Geometry::Segment:
                if (order == 1) {
                    return {Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
                }
                return {Vector3(-1.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)};
            case Geometry::Triangle:
                if (order == 1) {
                    return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)};
                }
                return {
                    Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0),
                    Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0)};
            case Geometry::Square:
                if (order == 1) {
                    return {Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0)};
                }
                return {
                    Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0),
                    Vector3(0.0, -1.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(-1.0, 0.0, 0.0),
                    Vector3(0.0, 0.0, 0.0)};
            case Geometry::Tetrahedron:
                if (order == 1) {
                    return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)};
                }
                return {
                    Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0),
                    Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0),
                    Vector3(0.0, 0.0, 0.5), Vector3(0.5, 0.0, 0.5), Vector3(0.0, 0.5, 0.5)};
            case Geometry::Cube:
                if (order == 1) {
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
            default:
                MPFEM_THROW(Exception, "H1FiniteElement unsupported geometry");
            }
        }

    } // namespace

    H1FiniteElement::H1FiniteElement(Geometry geom, int order)
        : geom_(geom)
        , order_(order)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(Exception, "H1FiniteElement: only order 1 and 2 supported");
        }
        switch (geom_) {
        case Geometry::Segment:
        case Geometry::Triangle:
        case Geometry::Square:
        case Geometry::Tetrahedron:
        case Geometry::Cube:
            return;
        default:
            MPFEM_THROW(Exception, "H1FiniteElement: unsupported geometry");
        }
    }

    int H1FiniteElement::numDofs() const
    {
        return h1NumDofs(geom_, order_);
    }

    DofLayout H1FiniteElement::dofLayout() const
    {
        return h1DofLayout(geom_, order_);
    }

    void H1FiniteElement::evalShape(const Vector3& xi, Matrix& shape) const
    {
        GeometryMapping::evalShape(geom_, order_, xi, shape);
    }

    void H1FiniteElement::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        GeometryMapping::evalDerivatives(geom_, order_, xi, derivatives);
    }

    std::vector<Vector3> H1FiniteElement::interpolationPoints() const
    {
        return buildInterpolationPoints(geom_, order_);
    }

    std::vector<int> H1FiniteElement::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(geom_, order_, faceIdx);
    }

} // namespace mpfem

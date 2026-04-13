#include "nd.hpp"

#include "core/exception.hpp"
#include "core/geometry.hpp"

namespace mpfem {

    namespace {

        int ndNumDofs(Geometry geom, int order)
        {
            if (order < 1 || order > 2) {
                MPFEM_THROW(ArgumentException, "NDFiniteElement supports order 1 and 2 only");
            }

            switch (geom) {
            case Geometry::Triangle:
                return order == 1 ? 3 : 8;
            case Geometry::Tetrahedron:
                return order == 1 ? 6 : 20;
            default:
                MPFEM_THROW(NotImplementedException, "NDFiniteElement supports triangle/tetrahedron only");
            }
        }

    } // namespace

    NDFiniteElement::NDFiniteElement(Geometry geom, int order)
        : geom_(geom), order_(order), numDofs_(ndNumDofs(geom, order))
    {
    }

    DofLayout NDFiniteElement::dofLayout() const
    {
        if (geom_ == Geometry::Triangle) {
            return order_ == 1 ? DofLayout {0, 1, 0, 0} : DofLayout {0, 2, 1, 0};
        }
        if (geom_ == Geometry::Tetrahedron) {
            return order_ == 1 ? DofLayout {0, 1, 0, 0} : DofLayout {0, 2, 2, 0};
        }
        MPFEM_THROW(NotImplementedException, "NDFiniteElement dof layout for geometry");
    }

    void NDFiniteElement::evalShape(const Vector3& xi, ShapeMatrix& shape) const
    {
        if (order_ != 1) {
            MPFEM_THROW(NotImplementedException, "NDFiniteElement::evalShape supports order 1 only");
        }

        const int d = geom::dim(geom_);
        shape.setZero(numDofs_, d);

        Real L[4] = {0.0, 0.0, 0.0, 0.0};
        Real dL[4][3] = {};

        if (geom_ == Geometry::Triangle) {
            const Real x = xi.x();
            const Real y = xi.y();
            L[0] = 1.0 - x - y;
            L[1] = x;
            L[2] = y;
            dL[0][0] = -1.0;
            dL[0][1] = -1.0;
            dL[1][0] = 1.0;
            dL[2][1] = 1.0;
        }
        else if (geom_ == Geometry::Tetrahedron) {
            const Real x = xi.x();
            const Real y = xi.y();
            const Real z = xi.z();
            L[0] = 1.0 - x - y - z;
            L[1] = x;
            L[2] = y;
            L[3] = z;
            dL[0][0] = -1.0;
            dL[0][1] = -1.0;
            dL[0][2] = -1.0;
            dL[1][0] = 1.0;
            dL[2][1] = 1.0;
            dL[3][2] = 1.0;
        }
        else {
            MPFEM_THROW(NotImplementedException, "NDFiniteElement::evalShape supports triangle/tetrahedron only");
        }

        for (int e = 0; e < numDofs_; ++e) {
            const auto [i, j] = geom::edgeVertices(geom_, e);
            for (int k = 0; k < d; ++k) {
                shape(e, k) = L[i] * dL[j][k] - L[j] * dL[i][k];
            }
        }
    }

    void NDFiniteElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        if (order_ != 1) {
            MPFEM_THROW(NotImplementedException, "NDFiniteElement::evalDerivatives supports order 1 only");
        }

        derivatives.setZero(numDofs_, geom::dim(geom_));

        if (geom_ == Geometry::Triangle) {
            const Real dL[3][2] = {
                {-1.0, -1.0},
                {1.0, 0.0},
                {0.0, 1.0}};
            for (int e = 0; e < numDofs_; ++e) {
                const auto [i, j] = geom::edgeVertices(geom_, e);
                const Real curl = 2.0 * (dL[i][0] * dL[j][1] - dL[i][1] * dL[j][0]);
                derivatives(e, 0) = curl;
            }
            return;
        }

        if (geom_ == Geometry::Tetrahedron) {
            const Real dL[4][3] = {
                {-1.0, -1.0, -1.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0}};
            for (int e = 0; e < numDofs_; ++e) {
                const auto [i, j] = geom::edgeVertices(geom_, e);
                derivatives(e, 0) = 2.0 * (dL[i][1] * dL[j][2] - dL[i][2] * dL[j][1]);
                derivatives(e, 1) = 2.0 * (dL[i][2] * dL[j][0] - dL[i][0] * dL[j][2]);
                derivatives(e, 2) = 2.0 * (dL[i][0] * dL[j][1] - dL[i][1] * dL[j][0]);
            }
            return;
        }

        static_cast<void>(xi);
        MPFEM_THROW(NotImplementedException, "NDFiniteElement::evalDerivatives supports triangle/tetrahedron only");
    }

    std::vector<Vector3> NDFiniteElement::interpolationPoints() const
    {
        if (order_ != 1) {
            MPFEM_THROW(NotImplementedException, "NDFiniteElement::interpolationPoints supports order 1 only");
        }

        std::vector<Vector3> points(static_cast<size_t>(numDofs_), Vector3::Zero());
        for (int e = 0; e < numDofs_; ++e) {
            Real coords[3] = {0.0, 0.0, 0.0};
            if (!geom::getEdgeMidpointCoords(geom_, e, coords)) {
                MPFEM_THROW(Exception, "NDFiniteElement::interpolationPoints failed to get edge midpoint");
            }
            points[static_cast<size_t>(e)] = Vector3(coords[0], coords[1], coords[2]);
        }
        return points;
    }

    std::vector<int> NDFiniteElement::faceDofs(int faceIdx) const
    {
        if (order_ != 1) {
            MPFEM_THROW(NotImplementedException, "NDFiniteElement::faceDofs supports order 1 only");
        }

        if (geom_ == Geometry::Triangle) {
            if (faceIdx != 0) {
                return {};
            }
            return {0, 1, 2};
        }

        if (geom_ == Geometry::Tetrahedron) {
            return geom::faceEdges(geom_, faceIdx);
        }

        return {};
    }

} // namespace mpfem

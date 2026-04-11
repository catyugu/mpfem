#include "h1.hpp"

#include "core/exception.hpp"
#include <array>

namespace mpfem {

    namespace {

        inline void ensureShapeStorage(const FiniteElement& fe, Matrix& shape)
        {
            if (shape.rows() != fe.numDofs() || shape.cols() != fe.vdim()) {
                shape.resize(fe.numDofs(), fe.vdim());
            }
            shape.setZero();
        }

        inline void ensureDerivativeStorage(const FiniteElement& fe, Matrix& derivatives)
        {
            if (derivatives.rows() != fe.numDofs() || derivatives.cols() != 3) {
                derivatives.resize(fe.numDofs(), 3);
            }
            derivatives.setZero();
        }

        inline void copyValuesToShape(std::span<const Real> values, Matrix& shape)
        {
            for (int i = 0; i < static_cast<int>(values.size()); ++i) {
                shape(i, 0) = values[i];
            }
        }

        inline void copyGradsToDerivativeMatrix(std::span<const Vector3> grads, Matrix& derivatives)
        {
            for (int i = 0; i < static_cast<int>(grads.size()); ++i) {
                derivatives(i, 0) = grads[i].x();
                derivatives(i, 1) = grads[i].y();
                derivatives(i, 2) = grads[i].z();
            }
        }

        template <typename FEType>
        std::vector<int> buildH1FaceDofs(const FEType& fe, int faceIdx)
        {
            std::vector<int> dofs;
            const Geometry g = fe.geometry();

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
            const int edgeDofs = fe.dofsPerEdge();
            const int edgeBase = geom::numCorners(g);

            for (int edgeIdx : faceEdges) {
                const int base = edgeBase + edgeIdx * edgeDofs;
                for (int j = 0; j < edgeDofs; ++j) {
                    dofs.push_back(base + j);
                }
            }

            if (g == Geometry::Cube) {
                const int faceBase = edgeBase + geom::numEdges(g) * edgeDofs;
                for (int j = 0; j < fe.dofsPerFace(); ++j) {
                    dofs.push_back(faceBase + faceIdx * fe.dofsPerFace() + j);
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
        ensureShapeStorage(*this, shape);

        std::array<Real, MaxDofsPerElement> values {};
        evalValuesImpl(xi, std::span<Real>(values.data(), numDofs()));
        copyValuesToShape(std::span<const Real>(values.data(), numDofs()), shape);
    }

    void H1SegmentShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        ensureDerivativeStorage(*this, derivatives);

        std::array<Vector3, MaxDofsPerElement> grads {};
        evalGradsImpl(xi, std::span<Vector3>(grads.data(), numDofs()));
        copyGradsToDerivativeMatrix(std::span<const Vector3>(grads.data(), numDofs()), derivatives);
    }

    std::vector<int> H1SegmentShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    void H1SegmentShape::evalValuesImpl(const Vector3& xi, std::span<Real> values) const
    {
        const Real x = xi.x();
        if (order_ == 1) {
            values[0] = 0.5 * (1.0 - x);
            values[1] = 0.5 * (1.0 + x);
            return;
        }

        values[0] = -0.5 * x * (1.0 - x);
        values[1] = 1.0 - x * x;
        values[2] = 0.5 * x * (1.0 + x);
    }

    void H1SegmentShape::evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const
    {
        const Real x = xi.x();
        if (order_ == 1) {
            grads[0] = Vector3(-0.5, 0.0, 0.0);
            grads[1] = Vector3(0.5, 0.0, 0.0);
            return;
        }

        grads[0] = Vector3(x - 0.5, 0.0, 0.0);
        grads[1] = Vector3(-2.0 * x, 0.0, 0.0);
        grads[2] = Vector3(x + 0.5, 0.0, 0.0);
    }

    std::vector<Vector3> H1SegmentShape::dofCoords() const
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
        ensureShapeStorage(*this, shape);

        std::array<Real, MaxDofsPerElement> values {};
        evalValuesImpl(xi, std::span<Real>(values.data(), numDofs()));
        copyValuesToShape(std::span<const Real>(values.data(), numDofs()), shape);
    }

    void H1TriangleShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        ensureDerivativeStorage(*this, derivatives);

        std::array<Vector3, MaxDofsPerElement> grads {};
        evalGradsImpl(xi, std::span<Vector3>(grads.data(), numDofs()));
        copyGradsToDerivativeMatrix(std::span<const Vector3>(grads.data(), numDofs()), derivatives);
    }

    std::vector<int> H1TriangleShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    void H1TriangleShape::evalValuesImpl(const Vector3& xi, std::span<Real> values) const
    {
        const Real xi1 = xi.x();
        const Real xi2 = xi.y();

        if (order_ == 1) {
            values[0] = 1.0 - xi1 - xi2;
            values[1] = xi1;
            values[2] = xi2;
            return;
        }

        values[0] = (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2);
        values[1] = xi1 * (2.0 * xi1 - 1.0);
        values[2] = xi2 * (2.0 * xi2 - 1.0);
        values[3] = 4.0 * xi1 * (1.0 - xi1 - xi2);
        values[4] = 4.0 * xi2 * (1.0 - xi1 - xi2);
        values[5] = 4.0 * xi1 * xi2;
    }

    void H1TriangleShape::evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const
    {
        const Real xi1 = xi.x();
        const Real xi2 = xi.y();

        if (order_ == 1) {
            grads[0] = Vector3(-1.0, -1.0, 0.0);
            grads[1] = Vector3(1.0, 0.0, 0.0);
            grads[2] = Vector3(0.0, 1.0, 0.0);
            return;
        }

        grads[0] = Vector3(4.0 * xi1 + 4.0 * xi2 - 3.0,
            4.0 * xi1 + 4.0 * xi2 - 3.0, 0.0);
        grads[1] = Vector3(4.0 * xi1 - 1.0, 0.0, 0.0);
        grads[2] = Vector3(0.0, 4.0 * xi2 - 1.0, 0.0);
        grads[3] = Vector3(4.0 - 8.0 * xi1 - 4.0 * xi2,
            -4.0 * xi1, 0.0);
        grads[4] = Vector3(-4.0 * xi2,
            4.0 - 4.0 * xi1 - 8.0 * xi2, 0.0);
        grads[5] = Vector3(4.0 * xi2, 4.0 * xi1, 0.0);
    }

    std::vector<Vector3> H1TriangleShape::dofCoords() const
    {
        if (order_ == 1) {
            return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)};
        }
        return {
            Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0),
            Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0)
        };
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
        ensureShapeStorage(*this, shape);

        std::array<Real, MaxDofsPerElement> values {};
        evalValuesImpl(xi, std::span<Real>(values.data(), numDofs()));
        copyValuesToShape(std::span<const Real>(values.data(), numDofs()), shape);
    }

    void H1SquareShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        ensureDerivativeStorage(*this, derivatives);

        std::array<Vector3, MaxDofsPerElement> grads {};
        evalGradsImpl(xi, std::span<Vector3>(grads.data(), numDofs()));
        copyGradsToDerivativeMatrix(std::span<const Vector3>(grads.data(), numDofs()), derivatives);
    }

    std::vector<int> H1SquareShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    void H1SquareShape::evalValuesImpl(const Vector3& xi, std::span<Real> values) const
    {
        const Real x = xi.x();
        const Real y = xi.y();

        if (order_ == 1) {
            const Real phi0x = 0.5 * (1.0 - x);
            const Real phi1x = 0.5 * (1.0 + x);
            const Real phi0y = 0.5 * (1.0 - y);
            const Real phi1y = 0.5 * (1.0 + y);

            values[0] = phi0x * phi0y;
            values[1] = phi1x * phi0y;
            values[2] = phi1x * phi1y;
            values[3] = phi0x * phi1y;
            return;
        }

        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);

        values[0] = px0 * py0;
        values[1] = px2 * py0;
        values[2] = px2 * py2;
        values[3] = px0 * py2;
        values[4] = px1 * py0;
        values[5] = px2 * py1;
        values[6] = px1 * py2;
        values[7] = px0 * py1;
        values[8] = px1 * py1;
    }

    void H1SquareShape::evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const
    {
        const Real x = xi.x();
        const Real y = xi.y();

        if (order_ == 1) {
            const Real phi0x = 0.5 * (1.0 - x);
            const Real phi1x = 0.5 * (1.0 + x);
            const Real phi0y = 0.5 * (1.0 - y);
            const Real phi1y = 0.5 * (1.0 + y);

            const Real dphi0x = -0.5;
            const Real dphi1x = 0.5;
            const Real dphi0y = -0.5;
            const Real dphi1y = 0.5;

            grads[0] = Vector3(dphi0x * phi0y, phi0x * dphi0y, 0.0);
            grads[1] = Vector3(dphi1x * phi0y, phi1x * dphi0y, 0.0);
            grads[2] = Vector3(dphi1x * phi1y, phi1x * dphi1y, 0.0);
            grads[3] = Vector3(dphi0x * phi1y, phi0x * dphi1y, 0.0);
            return;
        }

        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);

        const Real dpx0 = x - 0.5;
        const Real dpx1 = -2.0 * x;
        const Real dpx2 = x + 0.5;
        const Real dpy0 = y - 0.5;
        const Real dpy1 = -2.0 * y;
        const Real dpy2 = y + 0.5;

        grads[0] = Vector3(dpx0 * py0, px0 * dpy0, 0.0);
        grads[1] = Vector3(dpx2 * py0, px2 * dpy0, 0.0);
        grads[2] = Vector3(dpx2 * py2, px2 * dpy2, 0.0);
        grads[3] = Vector3(dpx0 * py2, px0 * dpy2, 0.0);
        grads[4] = Vector3(dpx1 * py0, px1 * dpy0, 0.0);
        grads[5] = Vector3(dpx2 * py1, px2 * dpy1, 0.0);
        grads[6] = Vector3(dpx1 * py2, px1 * dpy2, 0.0);
        grads[7] = Vector3(dpx0 * py1, px0 * dpy1, 0.0);
        grads[8] = Vector3(dpx1 * py1, px1 * dpy1, 0.0);
    }

    std::vector<Vector3> H1SquareShape::dofCoords() const
    {
        if (order_ == 1) {
            return {Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0)};
        }
        return {
            Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0),
            Vector3(0.0, -1.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(-1.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0)
        };
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
        ensureShapeStorage(*this, shape);

        std::array<Real, MaxDofsPerElement> values {};
        evalValuesImpl(xi, std::span<Real>(values.data(), numDofs()));
        copyValuesToShape(std::span<const Real>(values.data(), numDofs()), shape);
    }

    void H1TetrahedronShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        ensureDerivativeStorage(*this, derivatives);

        std::array<Vector3, MaxDofsPerElement> grads {};
        evalGradsImpl(xi, std::span<Vector3>(grads.data(), numDofs()));
        copyGradsToDerivativeMatrix(std::span<const Vector3>(grads.data(), numDofs()), derivatives);
    }

    std::vector<int> H1TetrahedronShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    void H1TetrahedronShape::evalValuesImpl(const Vector3& xi, std::span<Real> values) const
    {
        const Real xi1 = xi.x();
        const Real xi2 = xi.y();
        const Real xi3 = xi.z();

        if (order_ == 1) {
            values[0] = 1.0 - xi1 - xi2 - xi3;
            values[1] = xi1;
            values[2] = xi2;
            values[3] = xi3;
            return;
        }

        values[0] = (1.0 - xi1 - xi2 - xi3) * (1.0 - 2.0 * xi1 - 2.0 * xi2 - 2.0 * xi3);
        values[1] = xi1 * (2.0 * xi1 - 1.0);
        values[2] = xi2 * (2.0 * xi2 - 1.0);
        values[3] = xi3 * (2.0 * xi3 - 1.0);
        values[4] = 4.0 * xi1 * (1.0 - xi1 - xi2 - xi3);
        values[5] = 4.0 * xi2 * (1.0 - xi1 - xi2 - xi3);
        values[6] = 4.0 * xi1 * xi2;
        values[7] = 4.0 * xi3 * (1.0 - xi1 - xi2 - xi3);
        values[8] = 4.0 * xi1 * xi3;
        values[9] = 4.0 * xi2 * xi3;
    }

    void H1TetrahedronShape::evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const
    {
        const Real xi1 = xi.x();
        const Real xi2 = xi.y();
        const Real xi3 = xi.z();

        if (order_ == 1) {
            grads[0] = Vector3(-1.0, -1.0, -1.0);
            grads[1] = Vector3(1.0, 0.0, 0.0);
            grads[2] = Vector3(0.0, 1.0, 0.0);
            grads[3] = Vector3(0.0, 0.0, 1.0);
            return;
        }

        grads[0] = Vector3(4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0,
            4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0,
            4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0);
        grads[1] = Vector3(4.0 * xi1 - 1.0, 0.0, 0.0);
        grads[2] = Vector3(0.0, 4.0 * xi2 - 1.0, 0.0);
        grads[3] = Vector3(0.0, 0.0, 4.0 * xi3 - 1.0);
        grads[4] = Vector3(4.0 - 8.0 * xi1 - 4.0 * xi2 - 4.0 * xi3,
            -4.0 * xi1, -4.0 * xi1);
        grads[5] = Vector3(-4.0 * xi2,
            4.0 - 4.0 * xi1 - 8.0 * xi2 - 4.0 * xi3,
            -4.0 * xi2);
        grads[6] = Vector3(4.0 * xi2, 4.0 * xi1, 0.0);
        grads[7] = Vector3(-4.0 * xi3, -4.0 * xi3,
            4.0 - 4.0 * xi1 - 4.0 * xi2 - 8.0 * xi3);
        grads[8] = Vector3(4.0 * xi3, 0.0, 4.0 * xi1);
        grads[9] = Vector3(0.0, 4.0 * xi3, 4.0 * xi2);
    }

    std::vector<Vector3> H1TetrahedronShape::dofCoords() const
    {
        if (order_ == 1) {
            return {Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)};
        }
        return {
            Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0),
            Vector3(0.5, 0.0, 0.0), Vector3(0.0, 0.5, 0.0), Vector3(0.5, 0.5, 0.0),
            Vector3(0.0, 0.0, 0.5), Vector3(0.5, 0.0, 0.5), Vector3(0.0, 0.5, 0.5)
        };
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
        ensureShapeStorage(*this, shape);

        std::array<Real, MaxDofsPerElement> values {};
        evalValuesImpl(xi, std::span<Real>(values.data(), numDofs()));
        copyValuesToShape(std::span<const Real>(values.data(), numDofs()), shape);
    }

    void H1CubeShape::evalDerivatives(const Vector3& xi, Matrix& derivatives) const
    {
        ensureDerivativeStorage(*this, derivatives);

        std::array<Vector3, MaxDofsPerElement> grads {};
        evalGradsImpl(xi, std::span<Vector3>(grads.data(), numDofs()));
        copyGradsToDerivativeMatrix(std::span<const Vector3>(grads.data(), numDofs()), derivatives);
    }

    std::vector<int> H1CubeShape::faceDofs(int faceIdx) const
    {
        return buildH1FaceDofs(*this, faceIdx);
    }

    void H1CubeShape::evalValuesImpl(const Vector3& xi, std::span<Real> values) const
    {
        const Real x = xi.x();
        const Real y = xi.y();
        const Real z = xi.z();

        if (order_ == 1) {
            const Real phi0x = 0.5 * (1.0 - x);
            const Real phi1x = 0.5 * (1.0 + x);
            const Real phi0y = 0.5 * (1.0 - y);
            const Real phi1y = 0.5 * (1.0 + y);
            const Real phi0z = 0.5 * (1.0 - z);
            const Real phi1z = 0.5 * (1.0 + z);

            values[0] = phi0x * phi0y * phi0z;
            values[1] = phi1x * phi0y * phi0z;
            values[2] = phi1x * phi1y * phi0z;
            values[3] = phi0x * phi1y * phi0z;
            values[4] = phi0x * phi0y * phi1z;
            values[5] = phi1x * phi0y * phi1z;
            values[6] = phi1x * phi1y * phi1z;
            values[7] = phi0x * phi1y * phi1z;
            return;
        }

        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);
        const Real pz0 = -0.5 * z * (1.0 - z);
        const Real pz1 = 1.0 - z * z;
        const Real pz2 = 0.5 * z * (1.0 + z);

        values[0] = px0 * py0 * pz0;
        values[1] = px2 * py0 * pz0;
        values[2] = px2 * py2 * pz0;
        values[3] = px0 * py2 * pz0;
        values[4] = px0 * py0 * pz2;
        values[5] = px2 * py0 * pz2;
        values[6] = px2 * py2 * pz2;
        values[7] = px0 * py2 * pz2;

        values[8] = px1 * py0 * pz0;
        values[9] = px2 * py1 * pz0;
        values[10] = px1 * py2 * pz0;
        values[11] = px0 * py1 * pz0;
        values[12] = px0 * py0 * pz1;
        values[13] = px2 * py0 * pz1;
        values[14] = px2 * py2 * pz1;
        values[15] = px0 * py2 * pz1;
        values[16] = px1 * py0 * pz2;
        values[17] = px2 * py1 * pz2;
        values[18] = px1 * py2 * pz2;
        values[19] = px0 * py1 * pz2;

        values[20] = px1 * py1 * pz0;
        values[21] = px1 * py0 * pz1;
        values[22] = px2 * py1 * pz1;
        values[23] = px1 * py2 * pz1;
        values[24] = px0 * py1 * pz1;
        values[25] = px1 * py1 * pz2;

        values[26] = px1 * py1 * pz1;
    }

    void H1CubeShape::evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const
    {
        const Real x = xi.x();
        const Real y = xi.y();
        const Real z = xi.z();

        if (order_ == 1) {
            const Real phi0x = 0.5 * (1.0 - x);
            const Real phi1x = 0.5 * (1.0 + x);
            const Real phi0y = 0.5 * (1.0 - y);
            const Real phi1y = 0.5 * (1.0 + y);
            const Real phi0z = 0.5 * (1.0 - z);
            const Real phi1z = 0.5 * (1.0 + z);

            const Real dphi0x = -0.5;
            const Real dphi1x = 0.5;
            const Real dphi0y = -0.5;
            const Real dphi1y = 0.5;
            const Real dphi0z = -0.5;
            const Real dphi1z = 0.5;

            grads[0] = Vector3(dphi0x * phi0y * phi0z, phi0x * dphi0y * phi0z, phi0x * phi0y * dphi0z);
            grads[1] = Vector3(dphi1x * phi0y * phi0z, phi1x * dphi0y * phi0z, phi1x * phi0y * dphi0z);
            grads[2] = Vector3(dphi1x * phi1y * phi0z, phi1x * dphi1y * phi0z, phi1x * phi1y * dphi0z);
            grads[3] = Vector3(dphi0x * phi1y * phi0z, phi0x * dphi1y * phi0z, phi0x * phi1y * dphi0z);
            grads[4] = Vector3(dphi0x * phi0y * phi1z, phi0x * dphi0y * phi1z, phi0x * phi0y * dphi1z);
            grads[5] = Vector3(dphi1x * phi0y * phi1z, phi1x * dphi0y * phi1z, phi1x * phi0y * dphi1z);
            grads[6] = Vector3(dphi1x * phi1y * phi1z, phi1x * dphi1y * phi1z, phi1x * phi1y * dphi1z);
            grads[7] = Vector3(dphi0x * phi1y * phi1z, phi0x * dphi1y * phi1z, phi0x * phi1y * dphi1z);
            return;
        }

        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);
        const Real pz0 = -0.5 * z * (1.0 - z);
        const Real pz1 = 1.0 - z * z;
        const Real pz2 = 0.5 * z * (1.0 + z);

        const Real dpx0 = x - 0.5;
        const Real dpx1 = -2.0 * x;
        const Real dpx2 = x + 0.5;
        const Real dpy0 = y - 0.5;
        const Real dpy1 = -2.0 * y;
        const Real dpy2 = y + 0.5;
        const Real dpz0 = z - 0.5;
        const Real dpz1 = -2.0 * z;
        const Real dpz2 = z + 0.5;

        grads[0] = Vector3(dpx0 * py0 * pz0, px0 * dpy0 * pz0, px0 * py0 * dpz0);
        grads[1] = Vector3(dpx2 * py0 * pz0, px2 * dpy0 * pz0, px2 * py0 * dpz0);
        grads[2] = Vector3(dpx2 * py2 * pz0, px2 * dpy2 * pz0, px2 * py2 * dpz0);
        grads[3] = Vector3(dpx0 * py2 * pz0, px0 * dpy2 * pz0, px0 * py2 * dpz0);
        grads[4] = Vector3(dpx0 * py0 * pz2, px0 * dpy0 * pz2, px0 * py0 * dpz2);
        grads[5] = Vector3(dpx2 * py0 * pz2, px2 * dpy0 * pz2, px2 * py0 * dpz2);
        grads[6] = Vector3(dpx2 * py2 * pz2, px2 * dpy2 * pz2, px2 * py2 * dpz2);
        grads[7] = Vector3(dpx0 * py2 * pz2, px0 * dpy2 * pz2, px0 * py2 * dpz2);

        grads[8] = Vector3(dpx1 * py0 * pz0, px1 * dpy0 * pz0, px1 * py0 * dpz0);
        grads[9] = Vector3(dpx2 * py1 * pz0, px2 * dpy1 * pz0, px2 * py1 * dpz0);
        grads[10] = Vector3(dpx1 * py2 * pz0, px1 * dpy2 * pz0, px1 * py2 * dpz0);
        grads[11] = Vector3(dpx0 * py1 * pz0, px0 * dpy1 * pz0, px0 * py1 * dpz0);
        grads[12] = Vector3(dpx0 * py0 * pz1, px0 * dpy0 * pz1, px0 * py0 * dpz1);
        grads[13] = Vector3(dpx2 * py0 * pz1, px2 * dpy0 * pz1, px2 * py0 * dpz1);
        grads[14] = Vector3(dpx2 * py2 * pz1, px2 * dpy2 * pz1, px2 * py2 * dpz1);
        grads[15] = Vector3(dpx0 * py2 * pz1, px0 * dpy2 * pz1, px0 * py2 * dpz1);
        grads[16] = Vector3(dpx1 * py0 * pz2, px1 * dpy0 * pz2, px1 * py0 * dpz2);
        grads[17] = Vector3(dpx2 * py1 * pz2, px2 * dpy1 * pz2, px2 * py1 * dpz2);
        grads[18] = Vector3(dpx1 * py2 * pz2, px1 * dpy2 * pz2, px1 * py2 * dpz2);
        grads[19] = Vector3(dpx0 * py1 * pz2, px0 * dpy1 * pz2, px0 * py1 * dpz2);

        grads[20] = Vector3(dpx1 * py1 * pz0, px1 * dpy1 * pz0, px1 * py1 * dpz0);
        grads[21] = Vector3(dpx1 * py0 * pz1, px1 * dpy0 * pz1, px1 * py0 * dpz1);
        grads[22] = Vector3(dpx2 * py1 * pz1, px2 * dpy1 * pz1, px2 * py1 * dpz1);
        grads[23] = Vector3(dpx1 * py2 * pz1, px1 * dpy2 * pz1, px1 * py2 * dpz1);
        grads[24] = Vector3(dpx0 * py1 * pz1, px0 * dpy1 * pz1, px0 * py1 * dpz1);
        grads[25] = Vector3(dpx1 * py1 * pz2, px1 * dpy1 * pz2, px1 * py1 * dpz2);

        grads[26] = Vector3(dpx1 * py1 * pz1, px1 * dpy1 * pz1, px1 * py1 * dpz1);
    }

    std::vector<Vector3> H1CubeShape::dofCoords() const
    {
        if (order_ == 1) {
            return {
                Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), Vector3(1.0, 1.0, -1.0), Vector3(-1.0, 1.0, -1.0),
                Vector3(-1.0, -1.0, 1.0), Vector3(1.0, -1.0, 1.0), Vector3(1.0, 1.0, 1.0), Vector3(-1.0, 1.0, 1.0)
            };
        }

        return {
            Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), Vector3(1.0, 1.0, -1.0), Vector3(-1.0, 1.0, -1.0),
            Vector3(-1.0, -1.0, 1.0), Vector3(1.0, -1.0, 1.0), Vector3(1.0, 1.0, 1.0), Vector3(-1.0, 1.0, 1.0),
            Vector3(0.0, -1.0, -1.0), Vector3(1.0, 0.0, -1.0), Vector3(0.0, 1.0, -1.0), Vector3(-1.0, 0.0, -1.0),
            Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0),
            Vector3(0.0, -1.0, 1.0), Vector3(1.0, 0.0, 1.0), Vector3(0.0, 1.0, 1.0), Vector3(-1.0, 0.0, 1.0),
            Vector3(0.0, 0.0, -1.0), Vector3(0.0, -1.0, 0.0), Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 1.0, 0.0), Vector3(-1.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0),
            Vector3(0.0, 0.0, 0.0)
        };
    }

} // namespace mpfem

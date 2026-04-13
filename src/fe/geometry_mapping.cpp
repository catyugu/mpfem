#include "fe/geometry_mapping.hpp"

#include "core/exception.hpp"
#include "core/geometry.hpp"

namespace mpfem {

    void GeometryMapping::evalShape(Geometry geom, int order, const Vector3& xi, ShapeMatrix& shape)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(NotImplementedException, "GeometryMapping supports only order 1 and 2");
        }

        const Real x = xi.x();
        const Real y = xi.y();
        const Real z = xi.z();

        switch (geom) {
        case Geometry::Segment:
            if (order == 1) {
                shape.setZero();
                shape(0, 0) = 0.5 * (1.0 - x);
                shape(1, 0) = 0.5 * (1.0 + x);
                return;
            }
            shape.setZero();
            shape(0, 0) = -0.5 * x * (1.0 - x);
            shape(1, 0) = 1.0 - x * x;
            shape(2, 0) = 0.5 * x * (1.0 + x);
            return;

        case Geometry::Triangle:
            if (order == 1) {
                shape.setZero();
                shape(0, 0) = 1.0 - x - y;
                shape(1, 0) = x;
                shape(2, 0) = y;
                return;
            }
            shape.setZero();
            shape(0, 0) = (1.0 - x - y) * (1.0 - 2.0 * x - 2.0 * y);
            shape(1, 0) = x * (2.0 * x - 1.0);
            shape(2, 0) = y * (2.0 * y - 1.0);
            shape(3, 0) = 4.0 * x * (1.0 - x - y);
            shape(4, 0) = 4.0 * y * (1.0 - x - y);
            shape(5, 0) = 4.0 * x * y;
            return;

        case Geometry::Square:
            if (order == 1) {
                shape.setZero();
                const Real phi0x = 0.5 * (1.0 - x);
                const Real phi1x = 0.5 * (1.0 + x);
                const Real phi0y = 0.5 * (1.0 - y);
                const Real phi1y = 0.5 * (1.0 + y);
                shape(0, 0) = phi0x * phi0y;
                shape(1, 0) = phi1x * phi0y;
                shape(2, 0) = phi1x * phi1y;
                shape(3, 0) = phi0x * phi1y;
                return;
            }
            shape.setZero();
            {
                const Real px0 = -0.5 * x * (1.0 - x);
                const Real px1 = 1.0 - x * x;
                const Real px2 = 0.5 * x * (1.0 + x);
                const Real py0 = -0.5 * y * (1.0 - y);
                const Real py1 = 1.0 - y * y;
                const Real py2 = 0.5 * y * (1.0 + y);
                shape(0, 0) = px0 * py0;
                shape(1, 0) = px2 * py0;
                shape(2, 0) = px2 * py2;
                shape(3, 0) = px0 * py2;
                shape(4, 0) = px1 * py0;
                shape(5, 0) = px2 * py1;
                shape(6, 0) = px1 * py2;
                shape(7, 0) = px0 * py1;
                shape(8, 0) = px1 * py1;
            }
            return;

        case Geometry::Tetrahedron:
            if (order == 1) {
                shape.setZero();
                shape(0, 0) = 1.0 - x - y - z;
                shape(1, 0) = x;
                shape(2, 0) = y;
                shape(3, 0) = z;
                return;
            }
            shape.setZero();
            shape(0, 0) = (1.0 - x - y - z) * (1.0 - 2.0 * x - 2.0 * y - 2.0 * z);
            shape(1, 0) = x * (2.0 * x - 1.0);
            shape(2, 0) = y * (2.0 * y - 1.0);
            shape(3, 0) = z * (2.0 * z - 1.0);
            shape(4, 0) = 4.0 * x * (1.0 - x - y - z);
            shape(5, 0) = 4.0 * y * (1.0 - x - y - z);
            shape(6, 0) = 4.0 * x * y;
            shape(7, 0) = 4.0 * z * (1.0 - x - y - z);
            shape(8, 0) = 4.0 * x * z;
            shape(9, 0) = 4.0 * y * z;
            return;

        case Geometry::Cube:
            if (order == 1) {
                shape.setZero();
                const Real phi0x = 0.5 * (1.0 - x);
                const Real phi1x = 0.5 * (1.0 + x);
                const Real phi0y = 0.5 * (1.0 - y);
                const Real phi1y = 0.5 * (1.0 + y);
                const Real phi0z = 0.5 * (1.0 - z);
                const Real phi1z = 0.5 * (1.0 + z);
                shape(0, 0) = phi0x * phi0y * phi0z;
                shape(1, 0) = phi1x * phi0y * phi0z;
                shape(2, 0) = phi1x * phi1y * phi0z;
                shape(3, 0) = phi0x * phi1y * phi0z;
                shape(4, 0) = phi0x * phi0y * phi1z;
                shape(5, 0) = phi1x * phi0y * phi1z;
                shape(6, 0) = phi1x * phi1y * phi1z;
                shape(7, 0) = phi0x * phi1y * phi1z;
                return;
            }
            shape.setZero();
            {
                const Real px0 = -0.5 * x * (1.0 - x);
                const Real px1 = 1.0 - x * x;
                const Real px2 = 0.5 * x * (1.0 + x);
                const Real py0 = -0.5 * y * (1.0 - y);
                const Real py1 = 1.0 - y * y;
                const Real py2 = 0.5 * y * (1.0 + y);
                const Real pz0 = -0.5 * z * (1.0 - z);
                const Real pz1 = 1.0 - z * z;
                const Real pz2 = 0.5 * z * (1.0 + z);

                shape(0, 0) = px0 * py0 * pz0;
                shape(1, 0) = px2 * py0 * pz0;
                shape(2, 0) = px2 * py2 * pz0;
                shape(3, 0) = px0 * py2 * pz0;
                shape(4, 0) = px0 * py0 * pz2;
                shape(5, 0) = px2 * py0 * pz2;
                shape(6, 0) = px2 * py2 * pz2;
                shape(7, 0) = px0 * py2 * pz2;

                shape(8, 0) = px1 * py0 * pz0;
                shape(9, 0) = px2 * py1 * pz0;
                shape(10, 0) = px1 * py2 * pz0;
                shape(11, 0) = px0 * py1 * pz0;
                shape(12, 0) = px0 * py0 * pz1;
                shape(13, 0) = px2 * py0 * pz1;
                shape(14, 0) = px2 * py2 * pz1;
                shape(15, 0) = px0 * py2 * pz1;
                shape(16, 0) = px1 * py0 * pz2;
                shape(17, 0) = px2 * py1 * pz2;
                shape(18, 0) = px1 * py2 * pz2;
                shape(19, 0) = px0 * py1 * pz2;

                shape(20, 0) = px1 * py1 * pz0;
                shape(21, 0) = px1 * py0 * pz1;
                shape(22, 0) = px2 * py1 * pz1;
                shape(23, 0) = px1 * py2 * pz1;
                shape(24, 0) = px0 * py1 * pz1;
                shape(25, 0) = px1 * py1 * pz2;

                shape(26, 0) = px1 * py1 * pz1;
            }
            return;

        default:
            MPFEM_THROW(NotImplementedException, "GeometryMapping unsupported geometry");
        }
    }

    void GeometryMapping::evalDerivatives(Geometry geom, int order, const Vector3& xi, DerivMatrix& derivatives)
    {
        if (order < 1 || order > 2) {
            MPFEM_THROW(NotImplementedException, "GeometryMapping supports only order 1 and 2");
        }

        const Real x = xi.x();
        const Real y = xi.y();
        const Real z = xi.z();

        switch (geom) {
        case Geometry::Segment:
            if (order == 1) {
                derivatives.setZero();
                derivatives(0, 0) = -0.5;
                derivatives(1, 0) = 0.5;
                return;
            }
            derivatives.setZero();
            derivatives(0, 0) = x - 0.5;
            derivatives(1, 0) = -2.0 * x;
            derivatives(2, 0) = x + 0.5;
            return;

        case Geometry::Triangle:
            if (order == 1) {
                derivatives.setZero();
                derivatives(0, 0) = -1.0;
                derivatives(0, 1) = -1.0;
                derivatives(1, 0) = 1.0;
                derivatives(2, 1) = 1.0;
                return;
            }
            derivatives.setZero();
            derivatives(0, 0) = 4.0 * x + 4.0 * y - 3.0;
            derivatives(0, 1) = 4.0 * x + 4.0 * y - 3.0;
            derivatives(1, 0) = 4.0 * x - 1.0;
            derivatives(2, 1) = 4.0 * y - 1.0;
            derivatives(3, 0) = 4.0 - 8.0 * x - 4.0 * y;
            derivatives(3, 1) = -4.0 * x;
            derivatives(4, 0) = -4.0 * y;
            derivatives(4, 1) = 4.0 - 4.0 * x - 8.0 * y;
            derivatives(5, 0) = 4.0 * y;
            derivatives(5, 1) = 4.0 * x;
            return;

        case Geometry::Square:
            if (order == 1) {
                derivatives.setZero();
                const Real phi0x = 0.5 * (1.0 - x);
                const Real phi1x = 0.5 * (1.0 + x);
                const Real phi0y = 0.5 * (1.0 - y);
                const Real phi1y = 0.5 * (1.0 + y);
                derivatives(0, 0) = -0.5 * phi0y;
                derivatives(0, 1) = -0.5 * phi0x;
                derivatives(1, 0) = 0.5 * phi0y;
                derivatives(1, 1) = -0.5 * phi1x;
                derivatives(2, 0) = 0.5 * phi1y;
                derivatives(2, 1) = 0.5 * phi1x;
                derivatives(3, 0) = -0.5 * phi1y;
                derivatives(3, 1) = 0.5 * phi0x;
                return;
            }
            derivatives.setZero();
            {
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

                derivatives(0, 0) = dpx0 * py0;
                derivatives(0, 1) = px0 * dpy0;
                derivatives(1, 0) = dpx2 * py0;
                derivatives(1, 1) = px2 * dpy0;
                derivatives(2, 0) = dpx2 * py2;
                derivatives(2, 1) = px2 * dpy2;
                derivatives(3, 0) = dpx0 * py2;
                derivatives(3, 1) = px0 * dpy2;
                derivatives(4, 0) = dpx1 * py0;
                derivatives(4, 1) = px1 * dpy0;
                derivatives(5, 0) = dpx2 * py1;
                derivatives(5, 1) = px2 * dpy1;
                derivatives(6, 0) = dpx1 * py2;
                derivatives(6, 1) = px1 * dpy2;
                derivatives(7, 0) = dpx0 * py1;
                derivatives(7, 1) = px0 * dpy1;
                derivatives(8, 0) = dpx1 * py1;
                derivatives(8, 1) = px1 * dpy1;
            }
            return;

        case Geometry::Tetrahedron:
            if (order == 1) {
                derivatives.setZero();
                derivatives(0, 0) = -1.0;
                derivatives(0, 1) = -1.0;
                derivatives(0, 2) = -1.0;
                derivatives(1, 0) = 1.0;
                derivatives(2, 1) = 1.0;
                derivatives(3, 2) = 1.0;
                return;
            }
            derivatives.setZero();
            derivatives(0, 0) = 4.0 * x + 4.0 * y + 4.0 * z - 3.0;
            derivatives(0, 1) = 4.0 * x + 4.0 * y + 4.0 * z - 3.0;
            derivatives(0, 2) = 4.0 * x + 4.0 * y + 4.0 * z - 3.0;
            derivatives(1, 0) = 4.0 * x - 1.0;
            derivatives(2, 1) = 4.0 * y - 1.0;
            derivatives(3, 2) = 4.0 * z - 1.0;
            derivatives(4, 0) = 4.0 - 8.0 * x - 4.0 * y - 4.0 * z;
            derivatives(4, 1) = -4.0 * x;
            derivatives(4, 2) = -4.0 * x;
            derivatives(5, 0) = -4.0 * y;
            derivatives(5, 1) = 4.0 - 4.0 * x - 8.0 * y - 4.0 * z;
            derivatives(5, 2) = -4.0 * y;
            derivatives(6, 0) = 4.0 * y;
            derivatives(6, 1) = 4.0 * x;
            derivatives(7, 0) = -4.0 * z;
            derivatives(7, 1) = -4.0 * z;
            derivatives(7, 2) = 4.0 - 4.0 * x - 4.0 * y - 8.0 * z;
            derivatives(8, 0) = 4.0 * z;
            derivatives(8, 2) = 4.0 * x;
            derivatives(9, 1) = 4.0 * z;
            derivatives(9, 2) = 4.0 * y;
            return;

        case Geometry::Cube:
            if (order == 1) {
                derivatives.setZero();
                const Real phi0x = 0.5 * (1.0 - x);
                const Real phi1x = 0.5 * (1.0 + x);
                const Real phi0y = 0.5 * (1.0 - y);
                const Real phi1y = 0.5 * (1.0 + y);
                const Real phi0z = 0.5 * (1.0 - z);
                const Real phi1z = 0.5 * (1.0 + z);

                derivatives(0, 0) = -0.5 * phi0y * phi0z;
                derivatives(0, 1) = -0.5 * phi0x * phi0z;
                derivatives(0, 2) = -0.5 * phi0x * phi0y;

                derivatives(1, 0) = 0.5 * phi0y * phi0z;
                derivatives(1, 1) = -0.5 * phi1x * phi0z;
                derivatives(1, 2) = -0.5 * phi1x * phi0y;

                derivatives(2, 0) = 0.5 * phi1y * phi0z;
                derivatives(2, 1) = 0.5 * phi1x * phi0z;
                derivatives(2, 2) = -0.5 * phi1x * phi1y;

                derivatives(3, 0) = -0.5 * phi1y * phi0z;
                derivatives(3, 1) = 0.5 * phi0x * phi0z;
                derivatives(3, 2) = -0.5 * phi0x * phi1y;

                derivatives(4, 0) = -0.5 * phi0y * phi1z;
                derivatives(4, 1) = -0.5 * phi0x * phi1z;
                derivatives(4, 2) = 0.5 * phi0x * phi0y;

                derivatives(5, 0) = 0.5 * phi0y * phi1z;
                derivatives(5, 1) = -0.5 * phi1x * phi1z;
                derivatives(5, 2) = 0.5 * phi1x * phi0y;

                derivatives(6, 0) = 0.5 * phi1y * phi1z;
                derivatives(6, 1) = 0.5 * phi1x * phi1z;
                derivatives(6, 2) = 0.5 * phi1x * phi1y;

                derivatives(7, 0) = -0.5 * phi1y * phi1z;
                derivatives(7, 1) = 0.5 * phi0x * phi1z;
                derivatives(7, 2) = 0.5 * phi0x * phi1y;
                return;
            }
            derivatives.setZero();
            {
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

                derivatives(0, 0) = dpx0 * py0 * pz0;
                derivatives(0, 1) = px0 * dpy0 * pz0;
                derivatives(0, 2) = px0 * py0 * dpz0;
                derivatives(1, 0) = dpx2 * py0 * pz0;
                derivatives(1, 1) = px2 * dpy0 * pz0;
                derivatives(1, 2) = px2 * py0 * dpz0;
                derivatives(2, 0) = dpx2 * py2 * pz0;
                derivatives(2, 1) = px2 * dpy2 * pz0;
                derivatives(2, 2) = px2 * py2 * dpz0;
                derivatives(3, 0) = dpx0 * py2 * pz0;
                derivatives(3, 1) = px0 * dpy2 * pz0;
                derivatives(3, 2) = px0 * py2 * dpz0;
                derivatives(4, 0) = dpx0 * py0 * pz2;
                derivatives(4, 1) = px0 * dpy0 * pz2;
                derivatives(4, 2) = px0 * py0 * dpz2;
                derivatives(5, 0) = dpx2 * py0 * pz2;
                derivatives(5, 1) = px2 * dpy0 * pz2;
                derivatives(5, 2) = px2 * py0 * dpz2;
                derivatives(6, 0) = dpx2 * py2 * pz2;
                derivatives(6, 1) = px2 * dpy2 * pz2;
                derivatives(6, 2) = px2 * py2 * dpz2;
                derivatives(7, 0) = dpx0 * py2 * pz2;
                derivatives(7, 1) = px0 * dpy2 * pz2;
                derivatives(7, 2) = px0 * py2 * dpz2;

                derivatives(8, 0) = dpx1 * py0 * pz0;
                derivatives(8, 1) = px1 * dpy0 * pz0;
                derivatives(8, 2) = px1 * py0 * dpz0;
                derivatives(9, 0) = dpx2 * py1 * pz0;
                derivatives(9, 1) = px2 * dpy1 * pz0;
                derivatives(9, 2) = px2 * py1 * dpz0;
                derivatives(10, 0) = dpx1 * py2 * pz0;
                derivatives(10, 1) = px1 * dpy2 * pz0;
                derivatives(10, 2) = px1 * py2 * dpz0;
                derivatives(11, 0) = dpx0 * py1 * pz0;
                derivatives(11, 1) = px0 * dpy1 * pz0;
                derivatives(11, 2) = px0 * py1 * dpz0;
                derivatives(12, 0) = dpx0 * py0 * pz1;
                derivatives(12, 1) = px0 * dpy0 * pz1;
                derivatives(12, 2) = px0 * py0 * dpz1;
                derivatives(13, 0) = dpx2 * py0 * pz1;
                derivatives(13, 1) = px2 * dpy0 * pz1;
                derivatives(13, 2) = px2 * py0 * dpz1;
                derivatives(14, 0) = dpx2 * py2 * pz1;
                derivatives(14, 1) = px2 * dpy2 * pz1;
                derivatives(14, 2) = px2 * py2 * dpz1;
                derivatives(15, 0) = dpx0 * py2 * pz1;
                derivatives(15, 1) = px0 * dpy2 * pz1;
                derivatives(15, 2) = px0 * py2 * dpz1;
                derivatives(16, 0) = dpx1 * py0 * pz2;
                derivatives(16, 1) = px1 * dpy0 * pz2;
                derivatives(16, 2) = px1 * py0 * dpz2;
                derivatives(17, 0) = dpx2 * py1 * pz2;
                derivatives(17, 1) = px2 * dpy1 * pz2;
                derivatives(17, 2) = px2 * py1 * dpz2;
                derivatives(18, 0) = dpx1 * py2 * pz2;
                derivatives(18, 1) = px1 * dpy2 * pz2;
                derivatives(18, 2) = px1 * py2 * dpz2;
                derivatives(19, 0) = dpx0 * py1 * pz2;
                derivatives(19, 1) = px0 * dpy1 * pz2;
                derivatives(19, 2) = px0 * py1 * dpz2;

                derivatives(20, 0) = dpx1 * py1 * pz0;
                derivatives(20, 1) = px1 * dpy1 * pz0;
                derivatives(20, 2) = px1 * py1 * dpz0;
                derivatives(21, 0) = dpx1 * py0 * pz1;
                derivatives(21, 1) = px1 * dpy0 * pz1;
                derivatives(21, 2) = px1 * py0 * dpz1;
                derivatives(22, 0) = dpx2 * py1 * pz1;
                derivatives(22, 1) = px2 * dpy1 * pz1;
                derivatives(22, 2) = px2 * py1 * dpz1;
                derivatives(23, 0) = dpx1 * py2 * pz1;
                derivatives(23, 1) = px1 * dpy2 * pz1;
                derivatives(23, 2) = px1 * py2 * dpz1;
                derivatives(24, 0) = dpx0 * py1 * pz1;
                derivatives(24, 1) = px0 * dpy1 * pz1;
                derivatives(24, 2) = px0 * py1 * dpz1;
                derivatives(25, 0) = dpx1 * py1 * pz2;
                derivatives(25, 1) = px1 * dpy1 * pz2;
                derivatives(25, 2) = px1 * py1 * dpz2;

                derivatives(26, 0) = dpx1 * py1 * pz1;
                derivatives(26, 1) = px1 * dpy1 * pz1;
                derivatives(26, 2) = px1 * py1 * dpz1;
            }
            return;

        default:
            MPFEM_THROW(NotImplementedException, "GeometryMapping unsupported geometry");
        }
    }

} // namespace mpfem

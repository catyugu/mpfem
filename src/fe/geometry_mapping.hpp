#ifndef MPFEM_GEOMETRY_MAPPING_HPP
#define MPFEM_GEOMETRY_MAPPING_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"


namespace mpfem {

    class GeometryMapping {
    public:
        static void evalShape(Geometry geom, int order, const Vector3& xi, ShapeMatrix& shape);
        static void evalDerivatives(Geometry geom, int order, const Vector3& xi, DerivMatrix& derivatives);
    };

} // namespace mpfem

#endif // MPFEM_GEOMETRY_MAPPING_HPP

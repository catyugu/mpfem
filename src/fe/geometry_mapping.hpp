#ifndef MPFEM_GEOMETRY_MAPPING_HPP
#define MPFEM_GEOMETRY_MAPPING_HPP

#include "core/types.hpp"
#include "mesh/geometry.hpp"

namespace mpfem {

class GeometryMapping {
public:
    static void evalShape(Geometry geom, int order, const Vector3& xi, Matrix& shape);
    static void evalDerivatives(Geometry geom, int order, const Vector3& xi, Matrix& derivatives);
};

} // namespace mpfem

#endif // MPFEM_GEOMETRY_MAPPING_HPP

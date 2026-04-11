#include "fe/facet_element_transform.hpp"
#include <cmath>

namespace mpfem {

    Vector3 FacetElementTransform::normal()
    {
        const Matrix& J = jacobian();
        Vector3 n = Vector3::Zero();

        if (dim_ == 2) {
            // Surface in 3D: normal is cross product of columns
            Vector3 t1(J(0, 0), J(1, 0), J(2, 0));
            Vector3 t2(J(0, 1), J(1, 1), J(2, 1));
            n = t1.cross(t2).normalized();
        }
        else if (dim_ == 1) {
            // Line in 3D: ambiguous without more context, but usually we handle 2D mesh (dim 1) or 3D mesh (dim 2)
            // For 2D (line in 2D space), we could do something like n = (-ty, tx)
            // But ElementTransform always has 3D physical coordinates in MPFEM.
            // If it's a line in 3D, we need a reference vector to define a normal.
            // Let's stick to the 2D surface in 3D case for now as it's the most common.
            Vector3 t1(J(0, 0), J(1, 0), J(2, 0));
            // Just a placeholder for 1D - this needs proper logic if 1D-in-3D is used.
            n = t1.cross(Vector3::UnitZ()).normalized();
        }

        return n;
    }

} // namespace mpfem

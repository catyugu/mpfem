#include "nd.hpp"

#include "core/exception.hpp"

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
        static_cast<void>(xi);
        shape.setZero(numDofs_, geom::dim(geom_));
    }

    void NDFiniteElement::evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const
    {
        static_cast<void>(xi);
        derivatives.setZero(numDofs_, geom::dim(geom_));
    }

    std::vector<Vector3> NDFiniteElement::interpolationPoints() const
    {
        return std::vector<Vector3>(static_cast<size_t>(numDofs_), Vector3::Zero());
    }

    std::vector<int> NDFiniteElement::faceDofs(int faceIdx) const
    {
        static_cast<void>(faceIdx);
        return {};
    }

} // namespace mpfem

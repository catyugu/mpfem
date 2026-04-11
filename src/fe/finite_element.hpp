#ifndef MPFEM_FINITE_ELEMENT_HPP
#define MPFEM_FINITE_ELEMENT_HPP

#include "core/types.hpp"
#include "mesh/geometry.hpp"
#include <memory>
#include <vector>

namespace mpfem {

    enum class BasisType {
        H1,
        L2,
        ND,
        RT
    };

    class FiniteElement {
    public:
        virtual ~FiniteElement() = default;

        virtual BasisType basisType() const = 0;
        virtual Geometry geometry() const = 0;
        virtual int order() const = 0;
        virtual int numDofs() const = 0;
        virtual int vdim() const = 0;

        virtual int dofsPerVertex() const = 0;
        virtual int dofsPerEdge() const = 0;
        virtual int dofsPerFace() const = 0;
        virtual int dofsPerVolume() const = 0;

        virtual void evalShape(const Vector3& xi, Matrix& shape) const = 0;
        virtual void evalDerivatives(const Vector3& xi, Matrix& derivatives) const = 0;

        virtual std::vector<Vector3> dofCoords() const = 0;
        virtual std::vector<int> faceDofs(int faceIdx) const = 0;

        int dim() const { return geom::dim(geometry()); }

        static std::unique_ptr<FiniteElement> create(BasisType type, Geometry geom, int order);
    };

} // namespace mpfem

#endif // MPFEM_FINITE_ELEMENT_HPP

#include "finite_element.hpp"

#include "core/exception.hpp"
#include "h1.hpp"

namespace mpfem {

    std::unique_ptr<FiniteElement> FiniteElement::create(BasisType type, Geometry geom, int order)
    {
        if (type != BasisType::H1) {
            MPFEM_THROW(NotImplementedException, "Only H1 basis is implemented");
        }

        switch (geom) {
        case Geometry::Segment:
            return std::make_unique<H1SegmentShape>(order);
        case Geometry::Triangle:
            return std::make_unique<H1TriangleShape>(order);
        case Geometry::Square:
            return std::make_unique<H1SquareShape>(order);
        case Geometry::Tetrahedron:
            return std::make_unique<H1TetrahedronShape>(order);
        case Geometry::Cube:
            return std::make_unique<H1CubeShape>(order);
        default:
            MPFEM_THROW(Exception, "Unsupported geometry for finite element");
        }
    }

} // namespace mpfem

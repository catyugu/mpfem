#include "finite_element.hpp"

#include "core/exception.hpp"
#include "h1.hpp"

namespace mpfem {

    std::unique_ptr<FiniteElement> FiniteElement::create(BasisType type, Geometry geom, int order, int vdim)
    {
        switch (type) {
        case BasisType::H1:
            return std::make_unique<H1FiniteElement>(geom, order, vdim);
        case BasisType::ND:
        case BasisType::L2:
        case BasisType::RT:
            MPFEM_THROW(NotImplementedException, "Finite element basis type is not implemented");
        }
        MPFEM_THROW(NotImplementedException, "Unknown finite element basis type");
    }

} // namespace mpfem

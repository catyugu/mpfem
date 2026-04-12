#include "finite_element.hpp"

#include "core/exception.hpp"
#include "h1.hpp"

namespace mpfem {

    std::unique_ptr<FiniteElement> FiniteElement::create(BasisType type, Geometry geom, int order)
    {
        if (type != BasisType::H1) {
            MPFEM_THROW(NotImplementedException, "Only H1 basis is implemented");
        }

        return std::make_unique<H1FiniteElement>(geom, order);
    }

} // namespace mpfem

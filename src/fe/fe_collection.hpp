#ifndef MPFEM_FE_COLLECTION_HPP
#define MPFEM_FE_COLLECTION_HPP

#include "core/exception.hpp"
#include "core/geometry.hpp"
#include "reference_element.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace mpfem {

    /**
     * @brief Finite element collection managing reference elements for all geometry types.
     *
     * A FECollection provides ReferenceElement objects for different geometry types
     * with a common polynomial order.
     *
     * This is inspired by MFEM's FiniteElementCollection design.
     */
    class FECollection {
    public:
        virtual ~FECollection() = default;

        virtual const ReferenceElement* get(Geometry geom) const = 0;
        virtual int order() const = 0;
        virtual int vdim() const = 0;
        virtual std::string name() const = 0;

        int numDofs(Geometry geom) const
        {
            const auto* elem = get(geom);
            return elem ? elem->numDofs() : 0;
        }

        bool hasGeometry(Geometry geom) const
        {
            return get(geom) != nullptr;
        }
    };

    class H1Collection final : public FECollection {
    public:
        H1Collection(int order, int vdim = 1)
            : order_(order), vdim_(vdim)
        {
            if (order_ < 1 || order_ > 2) {
                MPFEM_THROW(ArgumentException, "H1Collection supports order 1 and 2 only");
            }
            if (vdim_ < 1) {
                MPFEM_THROW(ArgumentException, "H1Collection requires vdim >= 1");
            }

            elements_[Geometry::Segment] = std::make_unique<ReferenceElement>(Geometry::Segment, order_, BasisType::H1);
            elements_[Geometry::Triangle] = std::make_unique<ReferenceElement>(Geometry::Triangle, order_, BasisType::H1);
            elements_[Geometry::Square] = std::make_unique<ReferenceElement>(Geometry::Square, order_, BasisType::H1);
            elements_[Geometry::Tetrahedron] = std::make_unique<ReferenceElement>(Geometry::Tetrahedron, order_, BasisType::H1);
            elements_[Geometry::Cube] = std::make_unique<ReferenceElement>(Geometry::Cube, order_, BasisType::H1);
        }

        const ReferenceElement* get(Geometry geom) const override
        {
            auto it = elements_.find(geom);
            return it != elements_.end() ? it->second.get() : nullptr;
        }

        int order() const override { return order_; }
        int vdim() const override { return vdim_; }
        std::string name() const override { return "H1"; }

    private:
        int order_ = 1;
        int vdim_ = 1;
        std::unordered_map<Geometry, std::unique_ptr<ReferenceElement>> elements_;
    };

} // namespace mpfem

#endif // MPFEM_FE_COLLECTION_HPP

#ifndef MPFEM_FE_COLLECTION_HPP
#define MPFEM_FE_COLLECTION_HPP

#include "mesh/geometry.hpp"
#include "reference_element.hpp"
#include "core/exception.hpp"
#include <memory>
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
    /// FE type enumeration
    enum class Type {
        H1,     ///< Continuous Lagrange elements
        L2,     ///< Discontinuous elements (TODO)
        ND,     ///< Nedelec edge elements (TODO)
        RT      ///< Raviart-Thomas elements (TODO)
    };
    
    /// Default constructor
    FECollection() = default;
    
    /// Construct H1 collection with given order
    explicit FECollection(int order, Type type = Type::H1);
    
    // -------------------------------------------------------------------------
    // Access
    // -------------------------------------------------------------------------
    
    /// Get polynomial order
    int order() const { return order_; }
    
    /// Get FE type
    Type type() const { return type_; }
    
    /// Get reference element for a geometry type
    const ReferenceElement* get(Geometry geom) const {
        auto it = elements_.find(geom);
        return it != elements_.end() ? it->second.get() : nullptr;
    }
    
    /// Get number of dofs for a geometry type
    int numDofs(Geometry geom) const {
        auto* elem = get(geom);
        return elem ? elem->numDofs() : 0;
    }
    
    /// Check if collection has element for geometry
    bool hasGeometry(Geometry geom) const {
        return elements_.find(geom) != elements_.end();
    }
    
    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------
    
    /// Create H1 collection
    static std::unique_ptr<FECollection> createH1(int order) {
        return std::make_unique<FECollection>(order, Type::H1);
    }
    
private:
    void initialize();
    
    int order_ = 1;
    Type type_ = Type::H1;
    std::unordered_map<Geometry, std::unique_ptr<ReferenceElement>> elements_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline FECollection::FECollection(int order, Type type)
    : order_(order), type_(type) {
    initialize();
}

inline void FECollection::initialize() {
    elements_.clear();

    const auto addH1Elements = [this]() {
        elements_[Geometry::Segment] = std::make_unique<ReferenceElement>(Geometry::Segment, order_);
        elements_[Geometry::Triangle] = std::make_unique<ReferenceElement>(Geometry::Triangle, order_);
        elements_[Geometry::Square] = std::make_unique<ReferenceElement>(Geometry::Square, order_);
        elements_[Geometry::Tetrahedron] = std::make_unique<ReferenceElement>(Geometry::Tetrahedron, order_);
        elements_[Geometry::Cube] = std::make_unique<ReferenceElement>(Geometry::Cube, order_);
    };

    switch (type_) {
    case Type::H1:
        addH1Elements();
        return;
    case Type::L2:
    case Type::ND:
    case Type::RT:
        MPFEM_THROW(NotImplementedException, "FECollection type");
    }
}

}  // namespace mpfem

#endif  // MPFEM_FE_COLLECTION_HPP

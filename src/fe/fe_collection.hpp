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
    
    /// Get reference element (alternative name for MFEM compatibility)
    const ReferenceElement* FiniteElementForGeometry(Geometry geom) const {
        return get(geom);
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
    
    /// Get continuity type (for H1 elements)
    static int getContType(Type t) {
        switch (t) {
            case Type::H1: return 1;  // Continuous
            case Type::L2: return 0;  // Discontinuous
            case Type::ND: return 2;  // H(curl)
            case Type::RT: return 3;  // H(div)
            default: return -1;
        }
    }
    
    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------
    
    /// Create H1 collection
    static std::unique_ptr<FECollection> createH1(int order) {
        return std::make_unique<FECollection>(order, Type::H1);
    }
    
    /// Create collection from name (MFEM-style)
    static std::unique_ptr<FECollection> create(const std::string& name);
    
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
    // Create reference elements for all supported geometries
    if (type_ == Type::H1) {
        elements_[Geometry::Segment] = std::make_unique<ReferenceElement>(
            Geometry::Segment, order_);
        elements_[Geometry::Triangle] = std::make_unique<ReferenceElement>(
            Geometry::Triangle, order_);
        elements_[Geometry::Square] = std::make_unique<ReferenceElement>(
            Geometry::Square, order_);
        elements_[Geometry::Tetrahedron] = std::make_unique<ReferenceElement>(
            Geometry::Tetrahedron, order_);
        elements_[Geometry::Cube] = std::make_unique<ReferenceElement>(
            Geometry::Cube, order_);
    }
    // TODO: Implement L2, ND, RT
}

inline std::unique_ptr<FECollection> FECollection::create(const std::string& name) {
    // Parse name like "H1_1", "H1_2", "L2_1", etc.
    if (name.substr(0, 2) == "H1") {
        int order = 1;
        if (name.length() > 3) {
            order = std::stoi(name.substr(3));
        }
        return createH1(order);
    }
    // TODO: Parse other types (L2, ND, RT)
    
    MPFEM_THROW(Exception, 
        "FECollection::create: unsupported FE type '" + name + 
        "'. Supported types: H1_1, H1_2, etc.");
}

}  // namespace mpfem

#endif  // MPFEM_FE_COLLECTION_HPP

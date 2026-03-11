#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "mesh/geometry.hpp"
#include "mesh/element.hpp"
#include "shape_function.hpp"
#include "quadrature.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Reference element combining geometry, shape functions, and quadrature.
 * 
 * A ReferenceElement provides all the information needed for finite element
 * calculations on the reference (canonical) element.
 */
class ReferenceElement {
public:
    /// Default constructor
    ReferenceElement() = default;
    
    /// Construct from geometry and polynomial order
    ReferenceElement(Geometry geom, int order);
    
    // -------------------------------------------------------------------------
    // Geometry info
    // -------------------------------------------------------------------------
    
    /// Get geometry type
    Geometry geometry() const { return geometry_; }
    
    /// Get spatial dimension
    int dim() const { return geom::dim(geometry_); }
    
    /// Get polynomial order
    int order() const { return order_; }
    
    // -------------------------------------------------------------------------
    // Shape functions
    // -------------------------------------------------------------------------
    
    /// Get number of degrees of freedom (shape functions)
    int numDofs() const;
    
    /// Get shape function evaluator
    const ShapeFunction* shapeFunction() const { return shapeFunc_.get(); }
    
    /// Evaluate shape functions at reference coordinates
    ShapeValues evalShape(const Real* xi) const {
        return shapeFunc_->eval(xi);
    }
    
    /// Evaluate shape functions at integration point
    ShapeValues evalShape(const IntegrationPoint& ip) const {
        return shapeFunc_->eval(ip);
    }
    
    /// Get dof coordinates in reference element
    std::vector<std::vector<Real>> dofCoords() const {
        return shapeFunc_->dofCoords();
    }
    
    // -------------------------------------------------------------------------
    // Quadrature
    // -------------------------------------------------------------------------
    
    /// Get quadrature rule
    const QuadratureRule& quadrature() const { return quadrature_; }
    
    /// Get number of integration points
    int numQuadraturePoints() const { return quadrature_.size(); }
    
    /// Get integration point
    const IntegrationPoint& integrationPoint(int i) const {
        return quadrature_[i];
    }
    
    // -------------------------------------------------------------------------
    // Face information
    // -------------------------------------------------------------------------
    
    /// Get number of faces
    int numFaces() const { return geom::numFaces(geometry_); }
    
    /// Get geometry type of a face
    Geometry faceGeometry(int faceIdx) const {
        return geom::faceGeometry(geometry_, faceIdx);
    }
    
    /// Get face reference element (for boundary integration)
    const ReferenceElement* faceElement(int faceIdx) const {
        if (faceIdx >= 0 && faceIdx < static_cast<int>(faceElements_.size())) {
            return faceElements_[faceIdx].get();
        }
        return nullptr;
    }
    
    /// Get local vertex indices for a face
    std::vector<int> faceVertices(int faceIdx) const;
    
    /// Get local dof indices for a face
    std::vector<int> faceDofs(int faceIdx) const;
    
    // -------------------------------------------------------------------------
    // Edge information
    // -------------------------------------------------------------------------
    
    /// Get number of edges
    int numEdges() const { return geom::numEdges(geometry_); }
    
    /// Get local vertex indices for an edge
    std::pair<int, int> edgeVertices(int edgeIdx) const;
    
private:
    void initialize();
    
    Geometry geometry_ = Geometry::Invalid;
    int order_ = 1;
    std::unique_ptr<ShapeFunction> shapeFunc_;
    QuadratureRule quadrature_;
    std::vector<std::unique_ptr<ReferenceElement>> faceElements_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline ReferenceElement::ReferenceElement(Geometry geom, int order)
    : geometry_(geom), order_(order) {
    initialize();
}

inline int ReferenceElement::numDofs() const {
    return shapeFunc_ ? shapeFunc_->numDofs() : 0;
}

inline void ReferenceElement::initialize() {
    // Create shape function
    switch (geometry_) {
        case Geometry::Segment:
            shapeFunc_ = std::make_unique<H1SegmentShape>(order_);
            break;
        case Geometry::Triangle:
            shapeFunc_ = std::make_unique<H1TriangleShape>(order_);
            break;
        case Geometry::Square:
            shapeFunc_ = std::make_unique<H1SquareShape>(order_);
            break;
        case Geometry::Tetrahedron:
            shapeFunc_ = std::make_unique<H1TetrahedronShape>(order_);
            break;
        case Geometry::Cube:
            shapeFunc_ = std::make_unique<H1CubeShape>(order_);
            break;
        default:
            return;
    }
    
    // Create quadrature rule with order 2*order for exact integration
    quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));
    
    // Create face elements for volume elements
    if (dim() == 3) {
        faceElements_.reserve(numFaces());
        for (int f = 0; f < numFaces(); ++f) {
            faceElements_.push_back(
                std::make_unique<ReferenceElement>(faceGeometry(f), order_));
        }
    }
}

inline std::vector<int> ReferenceElement::faceVertices(int faceIdx) const {
    // Delegate to geom namespace
    return geom::faceVertices(geometry_, faceIdx);
}

inline std::vector<int> ReferenceElement::faceDofs(int faceIdx) const {
    // For linear elements, face dofs = face vertices
    // For higher order, need to include edge/face interior dofs
    return faceVertices(faceIdx);
}

inline std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const {
    // Delegate to geom namespace
    return geom::edgeVertices(geometry_, edgeIdx);
}

}  // namespace mpfem

#endif  // MPFEM_REFERENCE_ELEMENT_HPP
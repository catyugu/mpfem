#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "mesh/geometry.hpp"
#include "mesh/element.hpp"
#include "shape_function.hpp"
#include "quadrature.hpp"
#include "core/exception.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Reference element combining geometry, shape functions, and quadrature.
 * 
 * A ReferenceElement provides all the information needed for finite element
 * calculations on the reference (canonical) element.
 * 
 * Precomputes shape function values and gradients at all quadrature points
 * to avoid runtime memory allocation during assembly.
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
    // Precomputed shape function values at quadrature points (ZERO ALLOCATION)
    // -------------------------------------------------------------------------
    
    /// Get shape function value at quadrature point q, dof i
    Real shapeValue(int q, int i) const {
        return shapeValues_[q * numDofs_ + i];
    }
    
    /// Get shape function gradient at quadrature point q, dof i (reference coords)
    const Vector3& shapeGradient(int q, int i) const {
        return shapeGradients_[q * numDofs_ + i];
    }
    
    /// Get pointer to all shape values at quadrature point q (for vectorized access)
    const Real* shapeValuesAtQuad(int q) const {
        return &shapeValues_[q * numDofs_];
    }
    
    /// Get pointer to all shape gradients at quadrature point q
    const Vector3* shapeGradientsAtQuad(int q) const {
        return &shapeGradients_[q * numDofs_];
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
        if (faceIdx < 0 || faceIdx >= static_cast<int>(faceElements_.size())) {
            MPFEM_THROW(RangeException, 
                "ReferenceElement::faceElement: invalid face index " + 
                std::to_string(faceIdx) + ", num faces = " + 
                std::to_string(faceElements_.size()));
        }
        return faceElements_[faceIdx].get();
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
    void precomputeShapeValues();
    
    Geometry geometry_ = Geometry::Invalid;
    int order_ = 1;
    int numDofs_ = 0;  // Cached number of DOFs
    std::unique_ptr<ShapeFunction> shapeFunc_;
    QuadratureRule quadrature_;
    std::vector<std::unique_ptr<ReferenceElement>> faceElements_;
    
    // Precomputed shape function values at all quadrature points
    // Layout: [numQuadPoints * numDofs] - shapeValues_[q * numDofs + i]
    std::vector<Real> shapeValues_;
    
    // Precomputed shape function gradients (in reference coordinates) at all quadrature points
    // Layout: [numQuadPoints * numDofs] - shapeGradients_[q * numDofs + i]
    std::vector<Vector3> shapeGradients_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline ReferenceElement::ReferenceElement(Geometry geom, int order)
    : geometry_(geom), order_(order) {
    initialize();
}

inline int ReferenceElement::numDofs() const {
    return numDofs_;
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
    
    // Cache numDofs
    numDofs_ = shapeFunc_->numDofs();
    
    // Create quadrature rule with order 2*order for exact integration
    quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));
    
    // Precompute shape function values at all quadrature points
    precomputeShapeValues();
    
    // Create face elements for volume elements
    if (dim() == 3) {
        faceElements_.reserve(numFaces());
        for (int f = 0; f < numFaces(); ++f) {
            faceElements_.push_back(
                std::make_unique<ReferenceElement>(faceGeometry(f), order_));
        }
    }
}

inline void ReferenceElement::precomputeShapeValues() {
    if (!shapeFunc_ || numDofs_ == 0) return;
    
    const int nq = quadrature_.size();
    if (nq == 0) return;
    
    // Allocate storage (single allocation for all quadrature points)
    shapeValues_.resize(static_cast<size_t>(nq) * numDofs_);
    shapeGradients_.resize(static_cast<size_t>(nq) * numDofs_);
    
    // Precompute values at each quadrature point
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = quadrature_[q];
        ShapeValues sv = shapeFunc_->eval(ip);
        
        // Copy to contiguous storage
        Real* vals = &shapeValues_[static_cast<size_t>(q) * numDofs_];
        Vector3* grads = &shapeGradients_[static_cast<size_t>(q) * numDofs_];
        
        for (int i = 0; i < numDofs_; ++i) {
            vals[i] = sv.values[i];
            grads[i] = sv.gradients[i];
        }
    }
}

inline std::vector<int> ReferenceElement::faceVertices(int faceIdx) const {
    // Delegate to geom namespace
    return geom::faceVertices(geometry_, faceIdx);
}

inline std::vector<int> ReferenceElement::faceDofs(int faceIdx) const {
    std::vector<int> dofs;
    
    // Get corner vertex indices for this face
    std::vector<int> faceVerts = faceVertices(faceIdx);
    
    // Add corner DOFs (same as vertex indices for Lagrange elements)
    for (int v : faceVerts) {
        dofs.push_back(v);
    }
    
    // For order >= 2, add edge DOFs on the face
    if (order_ >= 2) {
        // Get edge indices for this face
        std::vector<int> faceEdgeIndices = geom::faceEdges(geometry_, faceIdx);
        
        // Edge DOFs start after corner DOFs
        int numElemCorners = geom::numCorners(geometry_);
        int numElemEdges = geom::numEdges(geometry_);
        
        for (int edgeIdx : faceEdgeIndices) {
            // Edge DOF index = numCorners + edgeIdx
            dofs.push_back(numElemCorners + edgeIdx);
        }
        
        // For tensor product elements (Cube), add face center DOF
        // Face center DOFs come after edge DOFs
        if (geometry_ == Geometry::Cube) {
            // Cube2: face center DOF index = numCorners + numEdges + faceIdx
            dofs.push_back(numElemCorners + numElemEdges + faceIdx);
        }
    }
    
    // TODO: For order >= 3, add additional face interior DOFs
    
    return dofs;
}

inline std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const {
    // Delegate to geom namespace
    return geom::edgeVertices(geometry_, edgeIdx);
}

}  // namespace mpfem

#endif  // MPFEM_REFERENCE_ELEMENT_HPP
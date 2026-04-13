#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "core/exception.hpp"
#include "core/geometry.hpp"
#include "finite_element.hpp"
#include "quadrature.hpp"
#include <memory>

namespace mpfem {

    /**
     * @brief Reference element combining geometry, FiniteElement basis, and quadrature.
     *
     * A ReferenceElement provides all the information needed for finite element
     * calculations on the reference (canonical) element.
     *
     * Precomputes basis values and derivatives at all quadrature points
     * to avoid runtime memory allocation during assembly.
     */
    class ReferenceElement {
    public:
        /// Default constructor
        ReferenceElement() = default;

        /// Construct from geometry and polynomial order
        ReferenceElement(Geometry geom, int order, BasisType basisType = BasisType::H1, int vdim = 1);

        // -------------------------------------------------------------------------
        // Geometry info
        // -------------------------------------------------------------------------

        /// Get geometry type
        Geometry geometry() const { return geometry_; }

        /// Get spatial dimension
        int dim() const { return geom::dim(geometry_); }

        /// Get polynomial order
        int order() const { return order_; }

        /// Get basis type
        BasisType basisType() const { return basisType_; }

        /// Get vector dimension used by this reference basis
        int vdim() const { return vdim_; }

        // -------------------------------------------------------------------------
        // Basis
        // -------------------------------------------------------------------------

        /// Get number of element degrees of freedom
        int numDofs() const;

        /// Get basis evaluator
        const FiniteElement& basis() const { return *basis_; }

        /// Get interpolation points in reference element
        std::vector<Vector3> interpolationPoints() const
        {
            return basis_->interpolationPoints();
        }

        // -------------------------------------------------------------------------
        // Precomputed basis values at quadrature points
        // -------------------------------------------------------------------------

        /// Get shape values matrix at quadrature point q: [numDofs x vdim]
        const ShapeMatrix& shapeValuesAtQuad(int q) const
        {
            return cachedShapeValues_[q];
        }

        /// Get derivatives matrix at quadrature point q: [numDofs x 3]
        const DerivMatrix& shapeDerivativesAtQuad(int q) const
        {
            return cachedDerivatives_[q];
        }

        // -------------------------------------------------------------------------
        // Quadrature
        // -------------------------------------------------------------------------

        /// Get quadrature rule
        const QuadratureRule& quadrature() const { return quadrature_; }

        /// Get number of integration points
        int numQuadraturePoints() const { return quadrature_.size(); }

        /// Get integration point
        const IntegrationPoint& integrationPoint(int i) const
        {
            return quadrature_[i];
        }

        // -------------------------------------------------------------------------
        // Face information
        // -------------------------------------------------------------------------

        /// Get number of faces
        int numFaces() const { return geom::numFaces(geometry_); }

        /// Get number of facets (dim-1 entities)
        int numFacets() const { return geom::numFacets(geometry_); }

        /// Get geometry type of a face
        Geometry faceGeometry(int faceIdx) const
        {
            return geom::faceGeometry(geometry_, faceIdx);
        }

        /// Get geometry type of a facet
        Geometry facetGeometry(int facetIdx) const
        {
            return geom::facetGeometry(geometry_, facetIdx);
        }

        /// Get local dof indices for a face
        std::vector<int> faceDofs(int faceIdx) const
        {
            return basis_->faceDofs(faceIdx);
        }

        /// Get local dof indices for a facet
        std::vector<int> facetDofs(int facetIdx) const
        {
            return basis_->facetDofs(facetIdx);
        }

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
        BasisType basisType_ = BasisType::H1;
        int vdim_ = 1;
        std::unique_ptr<FiniteElement> basis_;
        QuadratureRule quadrature_;
        std::vector<ShapeMatrix> cachedShapeValues_;
        std::vector<DerivMatrix> cachedDerivatives_;
    };

    // =============================================================================
    // Inline implementations
    // =============================================================================

    inline ReferenceElement::ReferenceElement(Geometry geom, int order, BasisType basisType, int vdim)
        : geometry_(geom), order_(order), basisType_(basisType), vdim_(vdim)
    {
        initialize();
    }

    inline int ReferenceElement::numDofs() const
    {
        return basis_ ? basis_->numDofs() : 0;
    }

    inline void ReferenceElement::initialize()
    {
        basis_ = FiniteElement::create(basisType_, geometry_, order_, vdim_);

        // Create quadrature rule with order 2*order for exact integration
        quadrature_ = quadrature::get(geometry_, std::max(1, 2 * order_));

        // Precompute basis values and derivatives at all quadrature points
        precomputeShapeValues();
    }

    inline void ReferenceElement::precomputeShapeValues()
    {
        if (!basis_)
            return;

        const int nq = quadrature_.size();
        if (nq == 0)
            return;

        cachedShapeValues_.resize(nq);
        cachedDerivatives_.resize(nq);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = quadrature_[q];
            basis_->evalShape(ip.getXi(), cachedShapeValues_[q]);
            basis_->evalDerivatives(ip.getXi(), cachedDerivatives_[q]);
        }
    }

    inline std::pair<int, int> ReferenceElement::edgeVertices(int edgeIdx) const
    {
        // Delegate to geom namespace
        return geom::edgeVertices(geometry_, edgeIdx);
    }

} // namespace mpfem

#endif // MPFEM_REFERENCE_ELEMENT_HPP
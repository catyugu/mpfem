#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "core/exception.hpp"
#include "core/geometry.hpp"
#include "quadrature.hpp"
#include <basix/finite-element.h>
#include <memory>
#include <vector>

namespace mpfem {

    /**
     * @brief Basis type enumeration for finite elements.
     *
     * Moved from deleted finite_element.hpp
     */
    enum class BasisType {
        H1,
        L2,
        ND,
        RT
    };

    /**
     * @brief Map type enumeration for Piola transformations.
     *
     * Moved from deleted finite_element.hpp
     */
    enum class MapType {
        VALUE, ///< Identity mapping (H1 elements)
        COVARIANT_PIOLA, ///< Covariant Piola mapping (ND elements)
        CONTRAVARIANT_PIOLA ///< Contravariant Piola mapping (RT elements)
    };

    /**
     * @brief DOF layout structure describing DOF distribution across topological entities.
     *
     * Moved from deleted finite_element.hpp
     */
    struct DofLayout {
        int numVertexDofs = 0;
        int numEdgeDofs = 0;
        int numFaceDofs = 0;
        int numVolumeDofs = 0;
    };

    /**
     * @brief Reference element combining geometry, BASIX FiniteElement basis, and quadrature.
     *
     * A ReferenceElement provides all the information needed for finite element
     * calculations on the reference (canonical) element.
     *
     * Precomputes basis values and derivatives at all quadrature points
     * to avoid runtime memory allocation during assembly.
     *
     * Uses basix::FiniteElement directly instead of a separate FiniteElement base class.
     * Implements CCW↔BASIX permutation mapping for tensor-product elements to align
     * DOF ordering with the CCW convention used by geom::edgeVertices() and geom::faceVertices().
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
        // Basis (BASIX)
        // -------------------------------------------------------------------------

        /// Get number of element degrees of freedom
        int numDofs() const;

        /// Get DOF layout (DOFs per vertex/edge/face/volume)
        DofLayout dofLayout() const;

        // -------------------------------------------------------------------------
        // Precomputed basis values at quadrature points
        // -------------------------------------------------------------------------

        /// Get shape values matrix at quadrature point q: [numDofs x vdim]
        const ShapeMatrix& shapeValuesAtQuad(int q) const
        {
            return cachedShapeValues_[q];
        }

        /// Get derivatives matrix at quadrature point q: [numDofs x dim*vdim]
        const DerivMatrix& shapeDerivativesAtQuad(int q) const
        {
            return cachedDerivatives_[q];
        }

        // -------------------------------------------------------------------------
        // Tabulation at arbitrary points (for GridFunction evaluation)
        // -------------------------------------------------------------------------

        /// Evaluate shape functions at arbitrary point xi: [numDofs x vdim]
        void evalShape(const Vector3& xi, ShapeMatrix& shape) const;

        /// Evaluate shape function derivatives at arbitrary point xi: [numDofs x dim*vdim]
        void evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const;

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

        /// Get number of faces (absolute dimension 2D)
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
        std::vector<int> faceDofs(int faceIdx) const;

        /// Get local dof indices for a facet
        std::vector<int> facetDofs(int facetIdx) const;

        // -------------------------------------------------------------------------
        // Edge information
        // -------------------------------------------------------------------------

        /// Get number of edges
        int numEdges() const { return geom::numEdges(geometry_); }

        /// Get local vertex indices for an edge
        std::pair<int, int> edgeVertices(int edgeIdx) const;

        /// Get local dof indices for an edge
        std::vector<int> edgeDofs(int edgeIdx) const;

    private:
        void initialize();
        void buildPermutation();
        void precomputeShapeValues();

        Geometry geometry_ = Geometry::Invalid;
        int order_ = 1;
        BasisType basisType_ = BasisType::H1;
        int vdim_ = 1;
        std::unique_ptr<basix::FiniteElement<double>> basixElement_; // Direct BASIX element
        QuadratureRule quadrature_;
        DofLayout dofLayout_; // Cached DOF layout from BASIX entity_dofs
        std::vector<int> ccwToBasix_; // [ccwIdx] = basixIdx
        std::vector<int> basixToCcw_; // [basixIdx] = ccwIdx
        std::vector<ShapeMatrix> cachedShapeValues_;
        std::vector<DerivMatrix> cachedDerivatives_;
    };

} // namespace mpfem

#endif // MPFEM_REFERENCE_ELEMENT_HPP

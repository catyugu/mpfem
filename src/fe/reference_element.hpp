#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "core/exception.hpp"
#include "core/geometry.hpp"
#include "quadrature.hpp"
#include <basix/finite-element.h>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace mpfem {

    enum class BasisType {
        H1,
        L2,
        ND,
        RT
    };

    enum class MapType {
        VALUE,
        COVARIANT_PIOLA,
        CONTRAVARIANT_PIOLA
    };

    struct DofLayout {
        int numVertexDofs = 0;
        int numEdgeDofs = 0;
        int numFaceDofs = 0;
        int numVolumeDofs = 0;
    };

    /**
     * @brief Reference element combining geometry, BASIX FiniteElement basis, and quadrature.
     *
     * Uses the Flyweight pattern - instances are globally cached and shared to avoid
     * the heavy cost of repeatedly tabulating shape functions and inferring topological
     * permutations. Access via static get() factory method.
     *
     * Completely delegates standard basis polynomial evaluation to BASIX.
     * Manages topological permutation to interface BASIX's UFC/tensor ordering
     * with the library's CCW ordering standard.
     */
    class ReferenceElement {
    public:
        // --- Flyweight Factory (thread-safe global cache) ---
        static const ReferenceElement* get(Geometry geom, int order, BasisType basisType = BasisType::H1, int vdim = 1);

        // Delete copy and move to enforce singleton behavior via cache
        ReferenceElement(const ReferenceElement&) = delete;
        ReferenceElement& operator=(const ReferenceElement&) = delete;

        Geometry geometry() const { return geometry_; }
        int dim() const { return geom::dim(geometry_); }
        int order() const { return order_; }
        BasisType basisType() const { return basisType_; }
        int vdim() const { return vdim_; }

        int numDofs() const;
        DofLayout dofLayout() const;

        const ShapeMatrix& shapeValuesAtQuad(int q) const { return cachedShapeValues_[q]; }
        const DerivMatrix& shapeDerivativesAtQuad(int q) const { return cachedDerivatives_[q]; }

        void evalShape(const Vector3& xi, ShapeMatrix& shape) const;
        void evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const;

        const QuadratureRule& quadrature() const { return quadrature_; }
        std::vector<Vector3> interpolationPoints() const;
        int numQuadraturePoints() const { return quadrature_.size(); }
        const IntegrationPoint& integrationPoint(int i) const { return quadrature_[i]; }

        int numFaces() const { return geom::numFaces(geometry_); }
        int numFacets() const { return geom::numFacets(geometry_); }
        Geometry faceGeometry(int faceIdx) const { return geom::faceGeometry(geometry_, faceIdx); }
        Geometry facetGeometry(int facetIdx) const { return geom::facetGeometry(geometry_, facetIdx); }

        std::vector<int> faceDofs(int faceIdx) const;
        std::vector<int> facetDofs(int facetIdx) const;

        int numEdges() const { return geom::numEdges(geometry_); }
        std::pair<int, int> edgeVertices(int edgeIdx) const;
        std::vector<int> edgeDofs(int edgeIdx) const;

    private:
        // Private constructor - only callable via get() factory
        ReferenceElement(Geometry geom, int order, BasisType basisType, int vdim);

        void initialize();
        void buildPermutation();
        void precomputeShapeValues();

        Geometry geometry_ = Geometry::Invalid;
        int order_ = 1;
        BasisType basisType_ = BasisType::H1;
        int vdim_ = 1;
        std::unique_ptr<basix::FiniteElement<double>> basixElement_;
        QuadratureRule quadrature_;
        DofLayout dofLayout_;

        // DOF Permutations: CCW <-> BASIX index mapping
        std::vector<int> ccwToBasix_; // [ccwIdx] = basixIdx
        std::vector<int> basixToCcw_; // [basixIdx] = ccwIdx

        // Precomputed entity DOFs mapped to CCW layout to ensure fast queries
        std::vector<std::vector<int>> edgeDofs_;
        std::vector<std::vector<int>> faceDofs_;

        std::vector<ShapeMatrix> cachedShapeValues_;
        std::vector<DerivMatrix> cachedDerivatives_;
    };

} // namespace mpfem

#endif // MPFEM_REFERENCE_ELEMENT_HPP
#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "core/types.hpp"
#include "fe/finite_element.hpp"
#include "fe/quadrature.hpp"
#include "mesh/geometry.hpp"
#include "mesh/mesh.hpp"
#include <Eigen/Dense>
#include <array>
#include <memory>

namespace mpfem {

    /**
     * @file element_transform.hpp
     * @brief Element transformation from reference to physical coordinates.
     *
     * Design: NO mutable members. All geometric quantities are computed
     * eagerly when setElement() or setIntegrationPoint() is called.
     *
     * Thread safety: Each thread should have its own transform instance.
     *
     * Memory optimization: Uses fixed-size stack arrays (MAX_NODES=27) to avoid
     * heap allocation during element iteration. Supports up to order-2 elements.
     *
     * **Geometric vs Field Order**:
     * - Geometric order (geomOrder_): From mesh element, used for coordinate transformation
     * - Field order: From FECollection, used for FiniteElement basis in ReferenceElement
     *
     * This separation enables:
     * - Subparametric: geometric order < field order (linear mesh, quadratic field)
     * - Superparametric: geometric order > field order (quadratic mesh, linear field)
     */

    class ElementTransform {
    public:
        enum ElementType { VOLUME = 1,
            BOUNDARY = 2 };

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        ElementTransform() = default;

        explicit ElementTransform(const Mesh* mesh, Index elemIdx,
            ElementType type = VOLUME);

        virtual ~ElementTransform() = default;

        // -------------------------------------------------------------------------
        // Setup (computes geometry info eagerly)
        // -------------------------------------------------------------------------

        virtual void setMesh(const Mesh* mesh);
        virtual void setElement(Index elemIdx);

        /// Set integration point (recomputes Jacobian and weight)
        void setIntegrationPoint(const Vector3& xi);
        void setIntegrationPoint(const IntegrationPoint& ip);

        // -------------------------------------------------------------------------
        // Geometry info
        // -------------------------------------------------------------------------

        Geometry geometry() const { return geometry_; }
        int dim() const { return dim_; }
        int spaceDim() const { return spaceDim_; }
        Index elementIndex() const { return elemIdx_; }
        ElementType elementType() const { return elemType_; }
        const Mesh* mesh() const { return mesh_; }
        Index attribute() const;
        int geometricOrder() const { return geomOrder_; }
        bool isCurved() const { return geomOrder_ >= 2; }

        // -------------------------------------------------------------------------
        // Transformation
        // -------------------------------------------------------------------------

        virtual void transform(const Vector3& xi, Real* x);
        void transform(const IntegrationPoint& ip, Vector3& x);

        // -------------------------------------------------------------------------
        // Jacobian and related quantities (computed in setIntegrationPoint)
        // -------------------------------------------------------------------------

        const Matrix& jacobian() const { return jacobian_; }
        const Matrix& invJacobian() const { return invJacobian_; }
        const Matrix& invJacobianT() const { return invJacobianT_; }
        Real detJ() const { return detJ_; }
        virtual Real weight() const { return weight_; }

        // -------------------------------------------------------------------------
        // Gradient transformation
        // -------------------------------------------------------------------------

        void transformGradient(const Real* refGrad, Real* physGrad) const;
        void transformGradient(const Vector3& refGrad, Vector3& physGrad) const;

        // -------------------------------------------------------------------------
        // Element nodes (fixed-size for performance)
        // -------------------------------------------------------------------------

        /// Get node coordinates (span view)
        std::span<const Vector3> nodes() const
        {
            return std::span<const Vector3>(nodesBuf_.data(), numNodes_);
        }

        /// Get number of nodes
        int numNodes() const { return numNodes_; }

        const IntegrationPoint& integrationPoint() const { return ip_; }

    protected:
        void computeGeometryInfo();
        void computeJacobianAtIP();

        // -------------------------------------------------------------------------
        // Member variables
        // -------------------------------------------------------------------------
        const Mesh* mesh_ = nullptr;
        Index elemIdx_ = InvalidIndex;
        ElementType elemType_ = VOLUME;

        Geometry geometry_ = Geometry::Invalid;
        int dim_ = 0;
        int spaceDim_ = 0;
        int geomOrder_ = 1;
        int numNodes_ = 0;

        // Geometric basis used only for coordinate mapping.
        std::unique_ptr<FiniteElement> geoBasis_;

        // Fixed-size buffers (no heap allocation)
        std::array<Vector3, MaxNodesPerElement> nodesBuf_;
        std::array<Index, MaxNodesPerElement> nodeIndicesBuf_;
        Matrix geoShapeValues_; // [numNodes x 1]
        Matrix geoShapeDerivatives_; // [numNodes x 3]

        IntegrationPoint ip_;

        // All computed eagerly (no lazy evaluation)
        Matrix jacobian_; // spaceDim x dim
        Matrix invJacobian_; // dim x spaceDim
        Matrix invJacobianT_; // spaceDim x dim
        Real detJ_ = 0.0;
        Real weight_ = 0.0;
    };

    // =============================================================================
    // Inline implementations
    // =============================================================================

    inline ElementTransform::ElementTransform(const Mesh* mesh, Index elemIdx, ElementType type)
        : mesh_(mesh), elemIdx_(elemIdx), elemType_(type)
    {
        computeGeometryInfo();
    }

    inline void ElementTransform::setIntegrationPoint(const IntegrationPoint& ip)
    {
        setIntegrationPoint(ip.getXi());
    }

    inline void ElementTransform::transform(const IntegrationPoint& ip, Vector3& x)
    {
        transform(ip.getXi(), x.data());
    }

    inline void ElementTransform::transformGradient(const Vector3& refGrad, Vector3& physGrad) const
    {
        Real rg[3] = {refGrad.x(), refGrad.y(), refGrad.z()};
        Real pg[3];
        transformGradient(rg, pg);
        physGrad = Vector3(pg[0], pg[1], pg[2]);
    }

} // namespace mpfem

#endif // MPFEM_ELEMENT_TRANSFORM_HPP

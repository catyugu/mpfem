#ifndef MPFEM_ELEMENT_TRANSFORM_HPP
#define MPFEM_ELEMENT_TRANSFORM_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"
#include "fe/geometry_mapping.hpp"
#include <Eigen/Geometry>
#include <array>
#include <span>

namespace mpfem {

    /**
     * @file element_transform.hpp
     * @brief Element transformation from reference to physical coordinates.
     *
     * Design: Decoupled from Mesh, Lazy Evaluation for performance.
     *
     * Thread safety: Each thread should have its own transform instance.
     */
    class ElementTransform {
    public:
        ElementTransform() = default;

        // -------------------------------------------------------------------------
        // Setup (Decoupled from Mesh)
        // -------------------------------------------------------------------------

        /**
         * @brief Bind element data to the transform.
         */
        void bindElement(Geometry geom, int geomOrder, Index attribute, Index elementId, std::span<const Vector3> nodes)
        {
            geometry_ = geom;
            geomOrder_ = geomOrder;
            attribute_ = attribute;
            elementId_ = elementId;
            dim_ = geom::dim(geometry_);

            numNodes_ = static_cast<int>(nodes.size());
            for (int i = 0; i < numNodes_; ++i) {
                nodesBuf_[i] = nodes[i];
            }
        }

        /**
         * @brief Set integration point (Eager Evaluation: computes all mapping quantities)
         */
        void setIntegrationPoint(const Vector3& xi)
        {
            ipXi_ = xi;
            computeJacobian();
            computeInverse();
        }

        // -------------------------------------------------------------------------
        // Eager Accessors (No branching)
        // -------------------------------------------------------------------------

        inline const Matrix3& jacobian() const { return jacobian_; }
        inline Real weight() const { return weight_; }
        inline const Matrix3& invJacobianT() const { return invJacobianT_; }
        inline const Matrix3& invJacobian() const { return invJacobian_; }
        inline Real detJ() const { return detJ_; }

        Vector3 normal() const
        {
            Vector3 n = Vector3::Zero();
            if (dim_ == 2) {
                const Vector3 t1(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
                const Vector3 t2(jacobian_(0, 1), jacobian_(1, 1), jacobian_(2, 1));
                n = t1.cross(t2).normalized();
            }
            return n;
        }

        void setFaceInfo(Index adjElemIdx, int localFaceIdx)
        {
            adjElemIdx_ = adjElemIdx;
            localFaceIdx_ = localFaceIdx;
        }

        Index adjacentElementIndex() const { return adjElemIdx_; }
        int localFaceIndex() const { return localFaceIdx_; }

        // -------------------------------------------------------------------------
        // Transformation
        // -------------------------------------------------------------------------

        Vector3 transform(const Vector3& xi) const;
        Vector3 transform(const IntegrationPoint& ip) const { return transform(ip.getXi()); }

        // -------------------------------------------------------------------------
        // Properties
        // -------------------------------------------------------------------------

        Geometry geometry() const { return geometry_; }
        int dim() const { return dim_; }
        Index attribute() const { return attribute_; }
        Index elementId() const { return elementId_; }
        int geomOrder() const { return geomOrder_; }
        int numNodes() const { return numNodes_; }
        std::span<const Vector3> nodes() const { return {nodesBuf_.data(), static_cast<size_t>(numNodes_)}; }
        const Vector3& ipXi() const { return ipXi_; }

    protected:
        void computeJacobian();
        void computeInverse();

        // Element data
        Geometry geometry_ = Geometry::Invalid;
        int geomOrder_ = 1;
        Index attribute_ = InvalidIndex;
        Index elementId_ = InvalidIndex;
        int dim_ = 0;
        int numNodes_ = 0;
        Index adjElemIdx_ = InvalidIndex;
        int localFaceIdx_ = -1;

        // Buffers
        std::array<Vector3, MaxNodesPerElement> nodesBuf_;
        DerivMatrix geoShapeDerivatives_;

        // State
        Vector3 ipXi_;

        // Computed values
        Matrix3 jacobian_ = Matrix3::Zero();
        Matrix3 invJacobian_ = Matrix3::Zero();
        Matrix3 invJacobianT_ = Matrix3::Zero();
        Real detJ_ = 0.0;
        Real weight_ = 0.0;

        // Batch buffers
        std::array<Vector3, MaxQuadraturePoints> batchRefPoints_;
        std::array<Vector3, MaxQuadraturePoints> batchPhysPoints_;
        std::array<ElementTransform*, MaxQuadraturePoints> batchTransforms_;
    };

} // namespace mpfem

#endif // MPFEM_ELEMENT_TRANSFORM_HPP

#include "fe/element_transform.hpp"
#include "core/exception.hpp"
#include "core/kernels.hpp"
#include <cmath>

namespace mpfem {

    void ElementTransform::computeJacobian()
    {
        // Ensure derivatives buffer is sized correctly (zero-allocation if already large enough)
        if (geoShapeDerivatives_.rows() != numNodes_ || geoShapeDerivatives_.cols() != 3) {
            geoShapeDerivatives_.resize(numNodes_, 3);
        }

        // Evaluate geometric shape derivatives at integration point.
        GeometryMapping::evalDerivatives(geometry_, geomOrder_, ipXi_, geoShapeDerivatives_);

        // Compute Jacobian: J = sum_i (x_i * grad_phi_i^T)
        // Note: Jacobian is spaceDim x dim. In MPFEM, spaceDim is always 3.
        jacobian_.setZero();
        for (int i = 0; i < numNodes_; ++i) {
            const Real gx = geoShapeDerivatives_(i, 0);
            const Real gy = geoShapeDerivatives_(i, 1);
            const Real gz = geoShapeDerivatives_(i, 2);
            for (int d = 0; d < 3; ++d) {
                jacobian_(d, 0) += nodesBuf_[i][d] * gx;
                if (dim_ > 1)
                    jacobian_(d, 1) += nodesBuf_[i][d] * gy;
                if (dim_ > 2)
                    jacobian_(d, 2) += nodesBuf_[i][d] * gz;
            }
        }

        // Compute weight and determinant
        if (dim_ == 3) {
            detJ_ = kernels::det3(jacobian_.data());
            weight_ = std::abs(detJ_);
        }
        else if (dim_ == 2) {
            const Vector3 t1 = jacobian_.col(0);
            const Vector3 t2 = jacobian_.col(1);
            detJ_ = t1.cross(t2).norm();
            weight_ = detJ_;
        }
        else if (dim_ == 1) {
            detJ_ = jacobian_.col(0).norm();
            weight_ = detJ_;
        }
    }

    void ElementTransform::computeInverse()
    {
        invJacobian_.setZero();
        invJacobianT_.setZero();

        if (dim_ == 3) {
            if (std::abs(detJ_) > 1e-15) {
                kernels::inverse3(jacobian_.data(), invJacobian_.data());
            }
        }
        else if (dim_ == 2) {
            const Matrix32 J = jacobian_.leftCols<2>();
            const Matrix2 JtJ = J.transpose() * J;
            if (JtJ.determinant() != 0.0) {
                const Matrix23 pseudo = JtJ.ldlt().solve(J.transpose());
                invJacobian_.topRows<2>() = pseudo;
            }
        }
        else if (dim_ == 1) {
            const Vector3 t = jacobian_.col(0);
            const Real denom = t.squaredNorm();
            if (denom > 1e-15) {
                invJacobian_.row(0) = t.transpose() / denom;
            }
        }

        invJacobianT_ = invJacobian_.transpose();
    }

    Vector3 ElementTransform::transform(const Vector3& xi) const
    {
        ShapeMatrix geoShapeValues;
        geoShapeValues.resize(numNodes_, 1);
        GeometryMapping::evalShape(geometry_, geomOrder_, xi, geoShapeValues);

        Vector3 x = Vector3::Zero();
        for (int d = 0; d < 3; ++d) {
            for (int i = 0; i < numNodes_; ++i) {
                x[d] += geoShapeValues(i, 0) * nodesBuf_[i][d];
            }
        }
        return x;
    }

} // namespace mpfem

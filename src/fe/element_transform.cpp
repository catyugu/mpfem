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
        jacobian_.setZero(3, dim_);
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
            // Boundary element in 3D: surface area weight
            Matrix JtJ = jacobian_.transpose() * jacobian_;
            detJ_ = std::sqrt(std::abs(JtJ.determinant()));
            weight_ = detJ_;
        }
        else if (dim_ == 1) {
            // Line element in 3D
            Matrix JtJ = jacobian_.transpose() * jacobian_;
            detJ_ = std::sqrt(std::abs(JtJ.determinant()));
            weight_ = detJ_;
        }
    }

    void ElementTransform::computeInverse()
    {
        invJacobian_.resize(dim_, 3);

        if (dim_ == 3) {
            if (std::abs(detJ_) > 1e-15) {
                kernels::inverse3(jacobian_.data(), invJacobian_.data());
            }
            else {
                invJacobian_.setZero();
            }
        }
        else {
            // Non-square case: Pseudo-inverse (J^T * J)^-1 * J^T
            Matrix JtJ = jacobian_.transpose() * jacobian_;
            invJacobian_ = JtJ.ldlt().solve(jacobian_.transpose());
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

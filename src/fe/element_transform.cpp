#include "fe/element_transform.hpp"
#include "core/exception.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <iostream>

namespace mpfem {

    void ElementTransform::setMesh(const Mesh* mesh)
    {
        mesh_ = mesh;
        computeGeometryInfo();
    }

    void ElementTransform::setElement(Index elemIdx)
    {
        elemIdx_ = elemIdx;
        computeGeometryInfo();
    }

    void ElementTransform::setIntegrationPoint(const Vector3& xi)
    {
        ip_.xi = xi[0];
        if (dim_ > 1)
            ip_.eta = xi[1];
        if (dim_ > 2)
            ip_.zeta = xi[2];
        computeJacobianAtIP();
    }

    Index ElementTransform::attribute() const
    {
        if (!mesh_) {
            MPFEM_THROW(Exception, "ElementTransform::attribute: mesh not set");
        }

        if (elemType_ == VOLUME) {
            if (elemIdx_ >= mesh_->numElements()) {
                MPFEM_THROW(RangeException,
                    "ElementTransform::attribute: invalid element index");
            }
            return mesh_->element(elemIdx_).attribute();
        }
        else {
            if (elemIdx_ >= mesh_->numBdrElements()) {
                MPFEM_THROW(RangeException,
                    "ElementTransform::attribute: invalid boundary element index");
            }
            return mesh_->bdrElement(elemIdx_).attribute();
        }
    }

    void ElementTransform::computeGeometryInfo()
    {
        if (!mesh_)
            return;

        const Element* elem = nullptr;
        if (elemType_ == VOLUME) {
            if (elemIdx_ < 0 || elemIdx_ >= static_cast<Index>(mesh_->numElements()))
                return;
            elem = &mesh_->element(elemIdx_);
        }
        else {
            if (elemIdx_ < 0 || elemIdx_ >= static_cast<Index>(mesh_->numBdrElements()))
                return;
            elem = &mesh_->bdrElement(elemIdx_);
        }

        geometry_ = elem->geometry();
        spaceDim_ = mesh_->dim();
        dim_ = geom::dim(geometry_);
        geomOrder_ = elem->order();

        // Get node coordinates - copy to fixed-size buffer
        const auto& vertexIndices = elem->vertices();
        numNodes_ = static_cast<int>(vertexIndices.size());

        for (int i = 0; i < numNodes_; ++i) {
            nodeIndicesBuf_[i] = vertexIndices[i];
            nodesBuf_[i] = mesh_->vertex(vertexIndices[i]).toVector();
        }

        // Pre-allocate matrices
        jacobian_.setZero(spaceDim_, dim_);
        invJacobian_.setZero(dim_, spaceDim_);
        invJacobianT_.setZero(spaceDim_, dim_);

        geoShapeDerivatives_.resize(numNodes_, 3);
    }

    void ElementTransform::computeJacobianAtIP()
    {
        // Evaluate geometric shape derivatives at integration point.
        GeometryMapping::evalDerivatives(geometry_, geomOrder_, ip_.getXi(), geoShapeDerivatives_);

        // Compute Jacobian: J = sum_i (x_i * grad_phi_i^T)
        jacobian_.setZero(spaceDim_, dim_);
        for (int i = 0; i < numNodes_; ++i) {
            const Real gx = geoShapeDerivatives_(i, 0);
            const Real gy = geoShapeDerivatives_(i, 1);
            const Real gz = geoShapeDerivatives_(i, 2);
            for (int d = 0; d < spaceDim_; ++d) {
                jacobian_(d, 0) += nodesBuf_[i][d] * gx;
                if (dim_ > 1) {
                    jacobian_(d, 1) += nodesBuf_[i][d] * gy;
                }
                if (dim_ > 2) {
                    jacobian_(d, 2) += nodesBuf_[i][d] * gz;
                }
            }
        }

        // Compute weight and inverse using optimized kernels
        if (dim_ == spaceDim_) {
            // 方阵情况：使用优化的行列式和逆矩阵计算
            if (dim_ == 2) {
                detJ_ = kernels::det2(jacobian_.data());
                weight_ = std::abs(detJ_);
                if (std::abs(detJ_) > 1e-15) {
                    kernels::inverse2(jacobian_.data(), invJacobian_.data());
                }
            }
            else if (dim_ == 3) {
                detJ_ = kernels::det3(jacobian_.data());
                weight_ = std::abs(detJ_);
                if (std::abs(detJ_) > 1e-15) {
                    kernels::inverse3(jacobian_.data(), invJacobian_.data());
                }
            }
            else {
                // 1D 或其他情况：使用 Eigen
                detJ_ = jacobian_.determinant();
                weight_ = std::abs(detJ_);
                if (std::abs(detJ_) > 1e-15) {
                    invJacobian_ = jacobian_.inverse();
                }
            }
        }
        else {
            // 非方阵情况（边界单元）：使用 Eigen
            Matrix JtJ = jacobian_.transpose() * jacobian_;
            weight_ = std::sqrt(std::abs(JtJ.determinant()));
            detJ_ = weight_;
            invJacobian_ = JtJ.ldlt().solve(jacobian_.transpose());
        }

        // Compute transpose
        invJacobianT_ = invJacobian_.transpose();
    }

    Vector3 ElementTransform::transform(const Vector3& xi)
    {
        // Evaluate geometric basis values for coordinate transformation.
        Matrix geoShapeValues;
        geoShapeValues.resize(numNodes_, 1);
        GeometryMapping::evalShape(geometry_, geomOrder_, xi, geoShapeValues);

        Vector3 x = Vector3::Zero();
        for (int d = 0; d < spaceDim_; ++d) {
            for (int i = 0; i < numNodes_; ++i) {
                x[d] += geoShapeValues(i, 0) * nodesBuf_[i][d];
            }
        }

        return x;
    }

} // namespace mpfem

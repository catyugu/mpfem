#include "grid_function.hpp"
#include "finite_element.hpp"

namespace mpfem {

    namespace {
        // Thread-local buffers to avoid repeated heap allocation
        thread_local Matrix t_shapeBuf;
        thread_local Matrix t_derivBuf;
        thread_local std::vector<Index> t_dofsBuf;

        inline void ensureCapacity(int nShape, int nElemDofs)
        {
            if (t_shapeBuf.rows() != nShape || t_shapeBuf.cols() != 1) {
                t_shapeBuf.resize(nShape, 1);
            }
            if (t_derivBuf.rows() != nShape || t_derivBuf.cols() != 3) {
                t_derivBuf.resize(nShape, 3);
            }
            if (static_cast<int>(t_dofsBuf.size()) < nElemDofs) {
                t_dofsBuf.resize(nElemDofs);
            }
        }
    } // namespace

    Real GridFunction::eval(Index elem, const Vector3& xi) const
    {
        if (!fes_)
            return 0.0;

        const ReferenceElement* ref = fes_->elementRefElement(elem);
        if (!ref)
            return 0.0;

        const FiniteElement& basis = ref->basis();
        const int nd = basis.numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);
        ensureCapacity(nd, totalDofs);

        basis.evalShape(xi, t_shapeBuf);
        fes_->getElementDofs(elem, std::span<Index> {t_dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Real val = 0.0;
        for (int i = 0; i < nd; ++i) {
            val += t_shapeBuf(i, 0) * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
        }
        return val;
    }

    Vector3 GridFunction::gradient(Index elem, const Vector3& xi, const Matrix3& invJacobianTranspose) const
    {
        if (!fes_)
            return Vector3::Zero();

        const ReferenceElement* ref = fes_->elementRefElement(elem);
        if (!ref)
            return Vector3::Zero();

        const FiniteElement& basis = ref->basis();
        const int nd = basis.numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);
        ensureCapacity(nd, totalDofs);

        basis.evalDerivatives(xi, t_derivBuf);
        fes_->getElementDofs(elem, std::span<Index> {t_dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Vector3 gRef = Vector3::Zero();
        for (int i = 0; i < nd; ++i) {
            gRef.x() += t_derivBuf(i, 0) * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.y() += t_derivBuf(i, 1) * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.z() += t_derivBuf(i, 2) * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
        }

        return invJacobianTranspose * gRef;
    }

} // namespace mpfem

#include "grid_function.hpp"
#include "shape_function.hpp"

namespace mpfem {

    namespace {
        // Thread-local buffers to avoid repeated heap allocation
        thread_local std::vector<Real> t_phiBuf;
        thread_local std::vector<Vector3> t_gradBuf;
        thread_local std::vector<Index> t_dofsBuf;

        inline void ensureCapacity(int nShape, int nElemDofs)
        {
            if (static_cast<int>(t_phiBuf.size()) < nShape) {
                t_phiBuf.resize(nShape);
            }
            if (static_cast<int>(t_gradBuf.size()) < nShape) {
                t_gradBuf.resize(nShape);
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

        const ShapeFunction* sf = ref->shapeFunction();
        const int nd = sf->numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);
        ensureCapacity(nd, totalDofs);

        sf->evalValues(xi, std::span<Real>(t_phiBuf.data(), nd));
        fes_->getElementDofs(elem, std::span<Index> {t_dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Real val = 0.0;
        for (int i = 0; i < nd; ++i) {
            val += t_phiBuf[i] * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
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

        const ShapeFunction* sf = ref->shapeFunction();
        const int nd = sf->numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);
        ensureCapacity(nd, totalDofs);

        sf->evalGrads(xi, std::span<Vector3>(t_gradBuf.data(), nd));
        fes_->getElementDofs(elem, std::span<Index> {t_dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Vector3 gRef = Vector3::Zero();
        for (int i = 0; i < nd; ++i) {
            gRef += t_gradBuf[i] * values_[t_dofsBuf[static_cast<size_t>(i) * vdim]];
        }

        return invJacobianTranspose * gRef;
    }

} // namespace mpfem

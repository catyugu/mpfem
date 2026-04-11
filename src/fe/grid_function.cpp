#include "grid_function.hpp"
#include "core/exception.hpp"
#include "finite_element.hpp"

#include <array>

namespace mpfem {

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

        if (nd > MaxDofsPerElement || totalDofs > MaxDofsPerElement) {
            MPFEM_THROW(Exception, "GridFunction::eval exceeds fixed stack buffer limits");
        }

        Matrix shapeBuf;
        std::array<Index, MaxDofsPerElement> dofsBuf {};

        basis.evalShape(xi, shapeBuf);
        fes_->getElementDofs(elem, std::span<Index> {dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Real val = 0.0;
        for (int i = 0; i < nd; ++i) {
            val += shapeBuf(i, 0) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
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

        if (nd > MaxDofsPerElement || totalDofs > MaxDofsPerElement) {
            MPFEM_THROW(Exception, "GridFunction::gradient exceeds fixed stack buffer limits");
        }

        Matrix derivBuf;
        std::array<Index, MaxDofsPerElement> dofsBuf {};

        basis.evalDerivatives(xi, derivBuf);
        fes_->getElementDofs(elem, std::span<Index> {dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Vector3 gRef = Vector3::Zero();
        for (int i = 0; i < nd; ++i) {
            gRef.x() += derivBuf(i, 0) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.y() += derivBuf(i, 1) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.z() += derivBuf(i, 2) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
        }

        return invJacobianTranspose * gRef;
    }

} // namespace mpfem

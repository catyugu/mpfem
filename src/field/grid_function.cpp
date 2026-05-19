#include "field/grid_function.hpp"
#include "core/exception.hpp"
#include "fe/element_transform.hpp"

#include <array>

namespace mpfem {

    void GridFunction::getElementValues(Index elem, std::span<Real> outValues) const
    {
        if (!fes_)
            return;

        const int totalDofs = fes_->numElementDofs(elem);
        if (totalDofs > MaxDofsPerElement) {
            MPFEM_THROW(Exception, "GridFunction::getElementValues exceeds fixed stack buffer limits");
        }
        if (static_cast<int>(outValues.size()) < totalDofs) {
            MPFEM_THROW(ArgumentException, "GridFunction::getElementValues output buffer is too small");
        }

        std::array<Index, MaxDofsPerElement> dofsBuf {};
        fes_->getElementDofs(elem, std::span<Index> {dofsBuf.data(), static_cast<size_t>(totalDofs)});
        for (int i = 0; i < totalDofs; ++i) {
            outValues[i] = values_[dofsBuf[i]];
        }
    }

    Real GridFunction::eval(Index elem, const ElementTransform& trans) const
    {
        if (!fes_)
            return 0.0;

        const ReferenceElement* ref = fes_->elementRefElement(elem);
        if (!ref)
            return 0.0;

        const int nd = ref->numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);

        if (nd > MaxDofsPerElement || totalDofs > MaxDofsPerElement) {
            MPFEM_THROW(Exception, "GridFunction::eval exceeds fixed stack buffer limits");
        }

        ShapeMatrix shapeBuf;
        std::array<Index, MaxDofsPerElement> dofsBuf {};

        ref->evalShape(trans.ipXi(), shapeBuf);
        fes_->getElementDofs(elem, std::span<Index> {dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Real val = 0.0;
        for (int i = 0; i < nd; ++i) {
            val += shapeBuf(i, 0) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
        }
        return val;
    }

    Vector3 GridFunction::gradient(Index elem, const ElementTransform& trans) const
    {
        if (!fes_)
            return Vector3::Zero();

        const ReferenceElement* ref = fes_->elementRefElement(elem);
        if (!ref)
            return Vector3::Zero();

        const int nd = ref->numDofs();
        const int vdim = fes_->vdim();
        const int totalDofs = fes_->numElementDofs(elem);

        if (nd > MaxDofsPerElement || totalDofs > MaxDofsPerElement) {
            MPFEM_THROW(Exception, "GridFunction::gradient exceeds fixed stack buffer limits");
        }

        DerivMatrix derivBuf;
        std::array<Index, MaxDofsPerElement> dofsBuf {};

        ref->evalDerivatives(trans.ipXi(), derivBuf);
        fes_->getElementDofs(elem, std::span<Index> {dofsBuf.data(), static_cast<size_t>(totalDofs)});

        Vector3 gRef = Vector3::Zero();
        for (int i = 0; i < nd; ++i) {
            gRef.x() += derivBuf(i, 0) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.y() += derivBuf(i, 1) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
            gRef.z() += derivBuf(i, 2) * values_[dofsBuf[static_cast<size_t>(i) * vdim]];
        }

        return trans.invJacobianT() * gRef;
    }

} // namespace mpfem

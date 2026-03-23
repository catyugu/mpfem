#include "grid_function.hpp"
#include "element_transform.hpp"
#include "shape_function.hpp"

namespace mpfem
{

    namespace
    {
        // Thread-local buffers to avoid repeated heap allocation
        thread_local std::vector<Real> t_phiBuf;
        thread_local std::vector<Vector3> t_gradBuf;
        thread_local std::vector<Index> t_dofsBuf;
    } // namespace

    Real GridFunction::eval(Index elem, const Real *xi) const
    {
        if (!fes_)
            return 0.0;

        const ReferenceElement *ref = fes_->elementRefElement(elem);
        if (!ref)
            return 0.0;

        const ShapeFunction *sf = ref->shapeFunction();
        int nd = sf->numDofs();

        // Reuse thread-local buffer
        if (static_cast<int>(t_phiBuf.size()) < nd)
        {
            t_phiBuf.resize(nd);
        }
        if (static_cast<int>(t_dofsBuf.size()) < nd * fes_->vdim())
        {
            t_dofsBuf.resize(nd * fes_->vdim());
        }

        sf->evalValues(xi, t_phiBuf.data());
        fes_->getElementDofs(elem, std::span<Index>{t_dofsBuf.data(), t_dofsBuf.size()});

        Real val = 0.0;
        for (int i = 0; i < nd; ++i)
        {
            val += t_phiBuf[i] * values_[t_dofsBuf[i]];
        }
        return val;
    }

    Vector3 GridFunction::gradient(Index elem, const Real *xi, ElementTransform &trans) const
    {
        if (!fes_)
            return Vector3::Zero();

        const ReferenceElement *ref = fes_->elementRefElement(elem);
        if (!ref)
            return Vector3::Zero();

        const ShapeFunction *sf = ref->shapeFunction();
        int nd = sf->numDofs();

        // Reuse thread-local buffer
        if (static_cast<int>(t_gradBuf.size()) < nd)
        {
            t_gradBuf.resize(nd);
        }
        if (static_cast<int>(t_dofsBuf.size()) < nd * fes_->vdim())
        {
            t_dofsBuf.resize(nd * fes_->vdim());
        }

        sf->evalGrads(xi, t_gradBuf.data());
        fes_->getElementDofs(elem, std::span<Index>{t_dofsBuf.data(), t_dofsBuf.size()});

        Vector3 gRef = Vector3::Zero();
        for (int i = 0; i < nd; ++i)
        {
            gRef += t_gradBuf[i] * values_[t_dofsBuf[i]];
        }

        trans.setIntegrationPoint(xi);
        return trans.invJacobianT() * gRef;
    }

} // namespace mpfem

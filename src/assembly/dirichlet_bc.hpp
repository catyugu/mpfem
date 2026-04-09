#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "expr/variable_graph.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include "core/sparse_matrix.hpp"
#include <array>
#include <map>
#include <vector>

namespace mpfem {

    inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
        const FESpace& fes, const Mesh& mesh,
        const std::map<int, const VariableNode*>& bcValues)
    {
        const Index numDofs = fes.numDofs();
        if (numDofs == 0)
            return;

        std::vector<Real> dofVals(numDofs, 0.0);
        std::vector<char> hasVal(numDofs, 0);
        std::vector<Index> eliminated;

        FacetElementTransform trans;
        trans.setMesh(&mesh);

        for (const auto& [bid, coef] : bcValues) {
            if (!fes.isExternalBoundaryId(bid))
                continue;

            for (Index b = 0; b < mesh.numBdrElements(); ++b) {
                if (mesh.bdrElement(b).attribute() != bid)
                    continue;

                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem)
                    continue;

                const auto& dofCoords = refElem->dofCoords();
                const int nd = refElem->numDofs();
                const int totalDofs = nd * fes.vdim();
                if (totalDofs > MaxVectorDofsPerBdrElement)
                    continue;

                std::array<Index, MaxVectorDofsPerBdrElement> dofs {};
                fes.getBdrElementDofs(b, std::span<Index> {dofs.data(), static_cast<size_t>(totalDofs)});

                trans.setBoundaryElement(b);

                for (int i = 0; i < nd; ++i) {
                    Index d = dofs[i];
                    if (d == InvalidIndex || hasVal[d])
                        continue;

                    Real xi[3] = {0.0, 0.0, 0.0};
                    for (size_t c = 0; c < dofCoords[i].size() && c < 3; ++c) {
                        xi[c] = dofCoords[i][c];
                    }

                    trans.setIntegrationPoint(xi);
                    Real value = 0.0;
                    if (coef) {
                        std::array<Vector3, 1> refPts {Vector3(xi[0], xi[1], xi[2])};
                        std::array<Vector3, 1> physPts;
                        const IntegrationPoint& ip = trans.integrationPoint();
                        trans.transform(ip, physPts[0]);
                        std::array<Matrix3, 1> invJTs {trans.invJacobianT()};
                        EvaluationContext ctx;
                        ctx.domainId = static_cast<int>(trans.attribute());
                        ctx.elementId = trans.elementIndex();
                        ctx.referencePoints = std::span<const Vector3>(refPts.data(), refPts.size());
                        ctx.physicalPoints = std::span<const Vector3>(physPts.data(), physPts.size());
                        ctx.invJacobianTransposes = std::span<const Matrix3>(invJTs.data(), invJTs.size());
                        std::array<TensorValue, 1> out {};
                        coef->evaluateBatch(ctx, std::span<TensorValue>(out.data(), out.size()));
                        value = out[0].asScalar();
                    }

                    dofVals[d] = value;
                    hasVal[d] = 1;
                    eliminated.push_back(d);
                }
            }
        }

        mat.eliminateRows(eliminated, dofVals, rhs);
        for (Index d : eliminated)
            sol(d) = dofVals[d];
    }

    inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
        const FESpace& fes, const Mesh& mesh,
        const std::map<int, Vector3>& bcValues,
        int vdim)
    {
        const Index numDofs = fes.numDofs();
        if (numDofs == 0)
            return;

        std::vector<Real> dofVals(numDofs, 0.0);
        std::vector<char> hasVal(numDofs, 0);
        std::vector<Index> eliminated;

        FacetElementTransform trans;
        trans.setMesh(&mesh);

        for (const auto& [bid, coef] : bcValues) {
            if (!fes.isExternalBoundaryId(bid))
                continue;

            for (Index b = 0; b < mesh.numBdrElements(); ++b) {
                if (mesh.bdrElement(b).attribute() != bid)
                    continue;

                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem)
                    continue;

                const auto& dofCoords = refElem->dofCoords();
                const int nd = refElem->numDofs();
                const int totalDofs = nd * vdim;
                if (totalDofs > MaxVectorDofsPerBdrElement)
                    continue;

                std::array<Index, MaxVectorDofsPerBdrElement> dofs {};
                fes.getBdrElementDofs(b, std::span<Index> {dofs.data(), static_cast<size_t>(totalDofs)});

                trans.setBoundaryElement(b);

                for (int i = 0; i < nd; ++i) {
                    Real xi[3] = {0.0, 0.0, 0.0};
                    for (size_t c = 0; c < dofCoords[i].size() && c < 3; ++c) {
                        xi[c] = dofCoords[i][c];
                    }

                    trans.setIntegrationPoint(xi);
                    const Vector3 disp = coef;

                    for (int c = 0; c < vdim; ++c) {
                        Index d = dofs[i * vdim + c];
                        if (d != InvalidIndex && !hasVal[d]) {
                            dofVals[d] = disp[c];
                            hasVal[d] = 1;
                            eliminated.push_back(d);
                        }
                    }
                }
            }
        }

        mat.eliminateRows(eliminated, dofVals, rhs);
        for (Index d : eliminated)
            sol(d) = dofVals[d];
    }

    inline void applyDirichletBCComponent(SparseMatrix& mat, Vector& rhs, Vector& sol,
        const FESpace& fes, const Mesh& mesh,
        const std::map<int, Real>& componentBCs,
        int vdim)
    {
        const Index numDofs = fes.numDofs();
        if (numDofs == 0)
            return;

        std::vector<Real> dofVals(numDofs, 0.0);
        std::vector<char> hasVal(numDofs, 0);
        std::vector<Index> eliminated;

        for (const auto& [key, val] : componentBCs) {
            int bid = key / vdim;
            int comp = key % vdim;

            if (!fes.isExternalBoundaryId(bid))
                continue;

            for (Index b = 0; b < mesh.numBdrElements(); ++b) {
                if (mesh.bdrElement(b).attribute() != bid)
                    continue;

                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem)
                    continue;

                const int nd = refElem->numDofs();
                const int totalDofs = nd * vdim;
                if (totalDofs > MaxVectorDofsPerBdrElement)
                    continue;

                std::array<Index, MaxVectorDofsPerBdrElement> dofs {};
                fes.getBdrElementDofs(b, std::span<Index> {dofs.data(), static_cast<size_t>(totalDofs)});

                for (int i = 0; i < nd; ++i) {
                    Index d = dofs[i * vdim + comp];
                    if (d != InvalidIndex && !hasVal[d]) {
                        dofVals[d] = val;
                        hasVal[d] = 1;
                        eliminated.push_back(d);
                    }
                }
            }
        }

        mat.eliminateRows(eliminated, dofVals, rhs);
        for (Index d : eliminated)
            sol(d) = dofVals[d];
    }

} // namespace mpfem

#endif // MPFEM_DIRICHLET_BC_HPP
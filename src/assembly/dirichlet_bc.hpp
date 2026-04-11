#ifndef MPFEM_DIRICHLET_BC_HPP
#define MPFEM_DIRICHLET_BC_HPP

#include "core/exception.hpp"
#include "core/sparse_matrix.hpp"
#include "core/types.hpp"
#include "expr/variable_graph.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"
#include <array>
#include <map>
#include <vector>

namespace mpfem {

    inline void applyDirichletBC(SparseMatrix& mat, Vector& rhs, Vector& sol,
        const FESpace& fes, const Mesh& mesh,
        const std::map<int, const VariableNode*>& bcValues,
        bool updateMatrix = true)
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
                    const int vdim = fes.vdim();

                    const Vector3& coord = dofCoords[i];
                    Real xi[3] = {coord.x(), coord.y(), coord.z()};

                    trans.setIntegrationPoint(coord);
                    std::array<Tensor, 1> out {};
                    if (coef) {
                        std::array<Vector3, 1> refPts {Vector3(xi[0], xi[1], xi[2])};
                        std::array<Vector3, 1> physPts;
                        const IntegrationPoint& ip = trans.integrationPoint();
                        physPts[0] = trans.transform(ip);
                        std::array<ElementTransform*, 1> transforms {&trans};
                        EvaluationContext ctx;
                        ctx.domainId = static_cast<int>(trans.attribute());
                        ctx.elementId = trans.elementIndex();
                        ctx.referencePoints = std::span<const Vector3>(refPts);
                        ctx.physicalPoints = std::span<const Vector3>(physPts);
                        ctx.transforms = std::span<ElementTransform* const>(transforms);
                        coef->evaluateBatch(ctx, std::span<Tensor>(out));
                    }

                    // Handle both scalar and vector-valued BCs
                    for (int c = 0; c < vdim; ++c) {
                        Index d = dofs[i * vdim + c]; // Global DOF index for component c
                        if (d == InvalidIndex || hasVal[d])
                            continue;

                        Real value = 0.0;
                        if (coef) {
                            if (out[0].isScalar()) {
                                value = out[0].asScalar();
                            }
                            else if (out[0].isVector()) {
                                value = out[0].asVector()(c);
                            }
                            else {
                                MPFEM_THROW(ArgumentException, "Dirichlet BC expects scalar or vector, got matrix");
                            }
                        }

                        dofVals[d] = value;
                        hasVal[d] = 1;
                        eliminated.push_back(d);
                    }
                }
            }
        }

        if (updateMatrix) {
            mat.eliminateRows(eliminated, dofVals, rhs);
        }
        else {
            mat.eliminateRhsOnly(eliminated, dofVals, rhs);
        }
        for (Index d : eliminated)
            sol(d) = dofVals[d];
    }

} // namespace mpfem

#endif // MPFEM_DIRICHLET_BC_HPP
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
#include <cmath>
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
        std::vector<Real> dofAccum(numDofs, 0.0);
        std::vector<Real> dofWeight(numDofs, 0.0);

        FacetElementTransform trans;

        for (const auto& [bid, coef] : bcValues) {
            if (!fes.isExternalBoundaryId(bid))
                continue;

            for (Index b = 0; b < mesh.numBdrElements(); ++b) {
                const Element& belem = mesh.bdrElement(b);
                if (belem.attribute() != bid)
                    continue;

                const ReferenceElement* refElem = fes.bdrElementRefElement(b);
                if (!refElem)
                    continue;

                const int nd = refElem->numDofs();
                const int totalDofs = nd * fes.vdim();
                if (totalDofs > MaxDofsPerBdrElement)
                    continue;

                std::array<Index, MaxDofsPerBdrElement> dofs {};
                fes.getBdrElementDofs(b, std::span<Index> {dofs.data(), static_cast<size_t>(totalDofs)});

                bindElementToTransform(trans, mesh, b, true);
                if (mesh.hasTopology()) {
                    Index faceIdx = mesh.getBoundaryFaceIndex(b);
                    if (faceIdx != InvalidIndex) {
                        const auto& faceInfo = mesh.getFaceInfo(faceIdx);
                        trans.setFaceInfo(faceInfo.elem1, faceInfo.localFace1);
                    }
                }

                Matrix localMass = Matrix::Zero(nd, nd);
                Matrix localRhs = Matrix::Zero(nd, fes.vdim());

                const QuadratureRule& rule = refElem->quadrature();
                for (int q = 0; q < rule.size(); ++q) {
                    const IntegrationPoint& ip = rule[q];
                    trans.setIntegrationPoint(ip.getXi());

                    const Real w = ip.weight * trans.weight();
                    const auto phi = refElem->shapeValuesAtQuad(q).col(0);
                    localMass.noalias() += w * (phi * phi.transpose());

                    std::array<Tensor, 1> out {};
                    if (coef) {
                        std::array<Vector3, 1> refPts {ip.getXi()};
                        std::array<Vector3, 1> physPts {trans.transform(ip)};
                        std::array<ElementTransform*, 1> transforms {&trans};
                        EvaluationContext ctx;
                        ctx.domainId = static_cast<int>(trans.attribute());
                        ctx.elementId = trans.elementId();
                        ctx.referencePoints = std::span<const Vector3>(refPts);
                        ctx.physicalPoints = std::span<const Vector3>(physPts);
                        ctx.transforms = std::span<ElementTransform* const>(transforms);
                        coef->evaluateBatch(ctx, std::span<Tensor>(out));
                    }

                    for (int c = 0; c < fes.vdim(); ++c) {
                        Real value = 0.0;
                        if (coef) {
                            if (out[0].isScalar()) {
                                value = out[0].asScalar();
                            }
                            else if (out[0].isVector()) {
                                value = (c < 3) ? out[0].asVector()(c) : 0.0;
                            }
                            else {
                                MPFEM_THROW(ArgumentException, "Dirichlet BC expects scalar or vector, got matrix");
                            }
                        }

                        localRhs.col(c).noalias() += w * value * phi;
                    }
                }

                Eigen::LDLT<Matrix> ldlt(localMass);
                if (ldlt.info() != Eigen::Success) {
                    continue;
                }

                for (int c = 0; c < fes.vdim(); ++c) {
                    const Vector coeffVec = ldlt.solve(localRhs.col(c));
                    for (int i = 0; i < nd; ++i) {
                        const Index d = dofs[i * fes.vdim() + c];
                        if (d == InvalidIndex) {
                            continue;
                        }

                        Real wi = std::abs(localMass(i, i));
                        if (wi <= 0.0) {
                            wi = 1.0;
                        }
                        dofAccum[d] += wi * coeffVec(i);
                        dofWeight[d] += wi;
                    }
                }
            }
        }

        std::vector<Index> eliminated;
        eliminated.reserve(numDofs);
        for (Index d = 0; d < numDofs; ++d) {
            if (dofWeight[d] <= 0.0) {
                continue;
            }
            dofVals[d] = dofAccum[d] / dofWeight[d];
            eliminated.push_back(d);
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
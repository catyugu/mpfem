#include "fe/fe_space.hpp"
#include "mesh/mesh.hpp"

#include <algorithm>

namespace mpfem {

    int FESpace::dim() const
    {
        return mesh_ ? mesh_->dim() : 0;
    }

    bool FESpace::isExternalBoundary(Index bdrElemIdx) const
    {
        return mesh_ ? mesh_->isExternalBoundary(bdrElemIdx) : true;
    }

    bool FESpace::isExternalBoundaryId(Index bdrId) const
    {
        return mesh_ ? mesh_->isExternalBoundaryId(bdrId) : true;
    }

    const ReferenceElement* FESpace::refElement(Geometry geom) const
    {
        return fec_ ? fec_->get(geom) : nullptr;
    }

    const ReferenceElement* FESpace::elementRefElement(Index elemIdx) const
    {
        if (!mesh_)
            MPFEM_THROW(Exception, "mesh not set");
        if (elemIdx >= mesh_->numElements())
            MPFEM_THROW(RangeException, "invalid element index");
        return refElement(mesh_->element(elemIdx).geometry);
    }

    const ReferenceElement* FESpace::bdrElementRefElement(Index bdrIdx) const
    {
        if (!mesh_)
            MPFEM_THROW(Exception, "mesh not set");
        if (bdrIdx >= mesh_->numBdrElements())
            MPFEM_THROW(RangeException, "invalid boundary element index");
        return refElement(mesh_->bdrElement(bdrIdx).geometry);
    }

    int FESpace::elementGeoOrder(Index elemIdx) const
    {
        return mesh_ ? mesh_->element(elemIdx).order : 1;
    }

    int FESpace::bdrElementGeoOrder(Index bdrIdx) const
    {
        return mesh_ ? mesh_->bdrElement(bdrIdx).order : 1;
    }

    void FESpace::getElementDofs(Index elemIdx, std::span<Index> dofs) const
    {
        if (!mesh_ || !fec_ || elemIdx >= mesh_->numElements())
            return;

        const Element elem = mesh_->element(elemIdx);
        const ReferenceElement* refElem = fec_->get(elem.geometry);
        if (!refElem)
            return;

        const int nd = refElem->numDofs();
        const int totalDofs = nd * vdim_;
        if (static_cast<int>(dofs.size()) < totalDofs)
            return;

        const Index base = elemIdx * maxDofsPerElem_;
        for (int i = 0; i < nd; ++i) {
            Index globalDof = elemDofs_[base + i];
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }

    void FESpace::getBdrElementDofs(Index bdrIdx, std::span<Index> dofs) const
    {
        if (!mesh_ || !fec_ || bdrIdx >= mesh_->numBdrElements())
            return;

        const Element bdrElem = mesh_->bdrElement(bdrIdx);
        const ReferenceElement* refElem = fec_->get(bdrElem.geometry);
        if (!refElem)
            return;

        const int nd = refElem->numDofs();
        const int totalDofs = nd * vdim_;
        if (static_cast<int>(dofs.size()) < totalDofs)
            return;

        const Index base = bdrIdx * maxDofsPerBdrElem_;
        for (int i = 0; i < nd; ++i) {
            Index globalDof = bdrElemDofs_[base + i];
            for (int c = 0; c < vdim_; ++c) {
                dofs[i * vdim_ + c] = globalDof * vdim_ + c;
            }
        }
    }

    int FESpace::numElementDofs(Index elemIdx) const
    {
        if (!mesh_ || !fec_ || elemIdx >= mesh_->numElements())
            return 0;
        const Element elem = mesh_->element(elemIdx);
        const ReferenceElement* refElem = fec_->get(elem.geometry);
        return refElem ? refElem->numDofs() * vdim_ : 0;
    }

    int FESpace::numBdrElementDofs(Index bdrIdx) const
    {
        if (!mesh_ || !fec_ || bdrIdx >= mesh_->numBdrElements())
            return 0;
        const Element bdrElem = mesh_->bdrElement(bdrIdx);
        const ReferenceElement* refElem = fec_->get(bdrElem.geometry);
        return refElem ? refElem->numDofs() * vdim_ : 0;
    }

    void FESpace::buildDofTable()
    {
        if (!mesh_ || !fec_) {
            MPFEM_THROW(Exception, "FESpace::buildDofTable requires both mesh and finite element collection");
        }

        if (!mesh_->hasTopology()) {
            MPFEM_THROW(Exception, "FESpace::buildDofTable requires mesh topology; call Mesh::buildTopology() first");
        }

        if (mesh_->numElements() == 0) {
            MPFEM_THROW(Exception, "FESpace::buildDofTable requires non-empty mesh");
        }

        maxDofsPerElem_ = 0;
        maxDofsPerBdrElem_ = 0;

        for (Index i = 0; i < mesh_->numElements(); ++i) {
            const ReferenceElement* refElem = fec_->get(mesh_->element(i).geometry);
            if (!refElem) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable missing volume reference element");
            }
            maxDofsPerElem_ = std::max(maxDofsPerElem_, refElem->numDofs());
        }

        for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
            const ReferenceElement* refElem = fec_->get(mesh_->bdrElement(i).geometry);
            if (!refElem) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable missing boundary reference element");
            }
            maxDofsPerBdrElem_ = std::max(maxDofsPerBdrElem_, refElem->numDofs());
        }

        const int meshDim = mesh_->dim();
        std::vector<int> vertexDofs(static_cast<size_t>(mesh_->numCornerVertices()), 0);
        std::vector<int> edgeDofs(static_cast<size_t>(mesh_->numEdges()), 0);
        std::vector<int> faceDofs;
        if (meshDim == 3) {
            faceDofs.assign(static_cast<size_t>(mesh_->numFaces()), 0);
        }
        std::vector<int> cellDofs(static_cast<size_t>(mesh_->numElements()), 0);

        for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
            const Element elem = mesh_->element(elemIdx);
            const ReferenceElement* refElem = fec_->get(elem.geometry);
            const DofLayout layout = refElem->basis().dofLayout();

            for (int i = 0; i < elem.numCorners(); ++i) {
                const Index vId = elem.vertex(i);
                const Index cornerId = mesh_->vertexToCornerIndex(vId);
                if (cornerId == InvalidIndex) {
                    MPFEM_THROW(Exception, "FESpace::buildDofTable non-corner vertex encountered");
                }
                const size_t idx = static_cast<size_t>(cornerId);
                vertexDofs[idx] = std::max(vertexDofs[idx], layout.numVertexDofs);
            }

            const auto elemEdges = mesh_->getElementEdges(elemIdx);
            for (Index edgeId : elemEdges) {
                const size_t idx = static_cast<size_t>(edgeId);
                edgeDofs[idx] = std::max(edgeDofs[idx], layout.numEdgeDofs);
            }

            if (meshDim == 3) {
                const auto elemFaces = mesh_->getElementFaces(elemIdx);
                for (Index faceId : elemFaces) {
                    const size_t idx = static_cast<size_t>(faceId);
                    faceDofs[idx] = std::max(faceDofs[idx], layout.numFaceDofs);
                }
                cellDofs[static_cast<size_t>(elemIdx)] = std::max(cellDofs[static_cast<size_t>(elemIdx)], layout.numVolumeDofs);
            }
            else {
                cellDofs[static_cast<size_t>(elemIdx)] = std::max(cellDofs[static_cast<size_t>(elemIdx)], layout.numFaceDofs);
            }
        }

        for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
            const Element bdrElem = mesh_->bdrElement(bdrIdx);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry);
            const DofLayout layout = refElem->basis().dofLayout();

            for (int i = 0; i < bdrElem.numCorners(); ++i) {
                const Index cornerId = mesh_->vertexToCornerIndex(bdrElem.vertex(i));
                if (cornerId == InvalidIndex) {
                    MPFEM_THROW(Exception, "FESpace::buildDofTable boundary corner vertex not found");
                }
                const size_t idx = static_cast<size_t>(cornerId);
                vertexDofs[idx] = std::max(vertexDofs[idx], layout.numVertexDofs);
            }

            for (int localEdge = 0; localEdge < bdrElem.numEdges(); ++localEdge) {
                const auto [v0, v1] = bdrElem.edgeVertices(localEdge);
                const Index edgeId = mesh_->edgeIndex(v0, v1);
                if (edgeId == InvalidIndex) {
                    MPFEM_THROW(Exception, "FESpace::buildDofTable boundary edge not found in topology");
                }
                const size_t idx = static_cast<size_t>(edgeId);
                edgeDofs[idx] = std::max(edgeDofs[idx], layout.numEdgeDofs);
            }

            if (meshDim == 3 && geom::dim(bdrElem.geometry) == 2) {
                const Index faceId = mesh_->getBoundaryFaceIndex(bdrIdx);
                if (faceId != InvalidIndex) {
                    const size_t idx = static_cast<size_t>(faceId);
                    faceDofs[idx] = std::max(faceDofs[idx], layout.numFaceDofs);
                }
            }
        }

        std::vector<Index> vertexOffset(vertexDofs.size(), 0);
        std::vector<Index> edgeOffset(edgeDofs.size(), 0);
        std::vector<Index> faceOffset(faceDofs.size(), 0);
        std::vector<Index> cellOffset(cellDofs.size(), 0);

        Index offset = 0;
        for (size_t i = 0; i < vertexDofs.size(); ++i) {
            vertexOffset[i] = offset;
            offset += vertexDofs[i];
        }
        for (size_t i = 0; i < edgeDofs.size(); ++i) {
            edgeOffset[i] = offset;
            offset += edgeDofs[i];
        }
        for (size_t i = 0; i < faceDofs.size(); ++i) {
            faceOffset[i] = offset;
            offset += faceDofs[i];
        }
        for (size_t i = 0; i < cellDofs.size(); ++i) {
            cellOffset[i] = offset;
            offset += cellDofs[i];
        }

        scalarNumDofs_ = offset;
        numDofs_ = scalarNumDofs_ * vdim_;

        elemDofs_.assign(mesh_->numElements() * maxDofsPerElem_, InvalidIndex);
        bdrElemDofs_.assign(mesh_->numBdrElements() * maxDofsPerBdrElem_, InvalidIndex);

        const auto mapVertexDof = [&](Index vertexId, int k) -> Index {
            const Index cornerId = mesh_->vertexToCornerIndex(vertexId);
            if (cornerId == InvalidIndex) {
                return InvalidIndex;
            }
            const size_t idx = static_cast<size_t>(cornerId);
            if (k < 0 || k >= vertexDofs[idx]) {
                return InvalidIndex;
            }
            return vertexOffset[idx] + k;
        };

        const auto mapEdgeDof = [&](Index edgeId, int k) -> Index {
            if (edgeId == InvalidIndex) {
                return InvalidIndex;
            }
            const size_t idx = static_cast<size_t>(edgeId);
            if (k < 0 || k >= edgeDofs[idx]) {
                return InvalidIndex;
            }
            return edgeOffset[idx] + k;
        };

        const auto mapFaceDof = [&](Index faceId, int k) -> Index {
            if (faceId == InvalidIndex) {
                return InvalidIndex;
            }
            const size_t idx = static_cast<size_t>(faceId);
            if (k < 0 || k >= faceDofs[idx]) {
                return InvalidIndex;
            }
            return faceOffset[idx] + k;
        };

        const auto mapCellDof = [&](Index elemIdx, int k) -> Index {
            const size_t idx = static_cast<size_t>(elemIdx);
            if (k < 0 || k >= cellDofs[idx]) {
                return InvalidIndex;
            }
            return cellOffset[idx] + k;
        };

        for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
            const Element elem = mesh_->element(elemIdx);
            const ReferenceElement* refElem = fec_->get(elem.geometry);
            const DofLayout layout = refElem->basis().dofLayout();

            const Index base = elemIdx * maxDofsPerElem_;
            int localDof = 0;

            for (int i = 0; i < elem.numCorners(); ++i) {
                const Index vId = elem.vertex(i);
                for (int k = 0; k < layout.numVertexDofs; ++k) {
                    const Index gdof = mapVertexDof(vId, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid vertex DOF mapping");
                    }
                    elemDofs_[base + localDof++] = gdof;
                }
            }

            const auto elemEdges = mesh_->getElementEdges(elemIdx);
            for (Index edgeId : elemEdges) {
                for (int k = 0; k < layout.numEdgeDofs; ++k) {
                    const Index gdof = mapEdgeDof(edgeId, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid edge DOF mapping");
                    }
                    elemDofs_[base + localDof++] = gdof;
                }
            }

            if (meshDim == 3 && layout.numFaceDofs > 0) {
                const auto elemFaces = mesh_->getElementFaces(elemIdx);
                for (Index faceId : elemFaces) {
                    for (int k = 0; k < layout.numFaceDofs; ++k) {
                        const Index gdof = mapFaceDof(faceId, k);
                        if (gdof == InvalidIndex) {
                            MPFEM_THROW(Exception, "FESpace::buildDofTable invalid face DOF mapping");
                        }
                        elemDofs_[base + localDof++] = gdof;
                    }
                }
            }

            if (meshDim == 2 && layout.numFaceDofs > 0) {
                for (int k = 0; k < layout.numFaceDofs; ++k) {
                    const Index gdof = mapCellDof(elemIdx, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid 2D interior DOF mapping");
                    }
                    elemDofs_[base + localDof++] = gdof;
                }
            }

            if (meshDim == 3 && layout.numVolumeDofs > 0) {
                for (int k = 0; k < layout.numVolumeDofs; ++k) {
                    const Index gdof = mapCellDof(elemIdx, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid volume DOF mapping");
                    }
                    elemDofs_[base + localDof++] = gdof;
                }
            }

            if (localDof != refElem->numDofs()) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable local DOF count mismatch on volume element");
            }
        }

        for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
            const Element bdrElem = mesh_->bdrElement(bdrIdx);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry);
            const DofLayout layout = refElem->basis().dofLayout();

            const Index base = bdrIdx * maxDofsPerBdrElem_;
            int localDof = 0;

            for (int i = 0; i < bdrElem.numCorners(); ++i) {
                const Index vId = bdrElem.vertex(i);
                for (int k = 0; k < layout.numVertexDofs; ++k) {
                    const Index gdof = mapVertexDof(vId, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid boundary vertex DOF mapping");
                    }
                    bdrElemDofs_[base + localDof++] = gdof;
                }
            }

            for (int localEdge = 0; localEdge < bdrElem.numEdges(); ++localEdge) {
                const auto [v0, v1] = bdrElem.edgeVertices(localEdge);
                const Index edgeId = mesh_->edgeIndex(v0, v1);
                for (int k = 0; k < layout.numEdgeDofs; ++k) {
                    const Index gdof = mapEdgeDof(edgeId, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid boundary edge DOF mapping");
                    }
                    bdrElemDofs_[base + localDof++] = gdof;
                }
            }

            if (meshDim == 3 && geom::dim(bdrElem.geometry) == 2 && layout.numFaceDofs > 0) {
                const Index faceId = mesh_->getBoundaryFaceIndex(bdrIdx);
                for (int k = 0; k < layout.numFaceDofs; ++k) {
                    const Index gdof = mapFaceDof(faceId, k);
                    if (gdof == InvalidIndex) {
                        MPFEM_THROW(Exception, "FESpace::buildDofTable invalid boundary face DOF mapping");
                    }
                    bdrElemDofs_[base + localDof++] = gdof;
                }
            }

            if (localDof != refElem->numDofs()) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable local DOF count mismatch on boundary element");
            }
        }
    }

} // namespace mpfem

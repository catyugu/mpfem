#include "fe/fe_space.hpp"
#include "mesh/element.hpp"
#include "mesh/mesh.hpp"

namespace mpfem {

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
            const Element& elem = mesh_->element(i);
            const ReferenceElement* refElem = fec_->get(elem.geometry());
            if (refElem) {
                maxDofsPerElem_ = std::max(maxDofsPerElem_, refElem->numDofs());
            }
        }

        for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
            const Element& bdrElem = mesh_->bdrElement(i);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
            if (refElem) {
                maxDofsPerBdrElem_ = std::max(maxDofsPerBdrElem_, refElem->numDofs());
            }
        }

        const Geometry meshGeom = mesh_->element(0).geometry();
        for (Index i = 1; i < mesh_->numElements(); ++i) {
            if (mesh_->element(i).geometry() != meshGeom) {
                MPFEM_THROW(NotImplementedException, "FESpace mixed volume geometries are not supported");
            }
        }

        const ReferenceElement* meshRefElem = fec_->get(meshGeom);
        if (!meshRefElem) {
            MPFEM_THROW(Exception, "FESpace::buildDofTable missing reference element");
        }

        const DofLayout layout = meshRefElem->basis().dofLayout();
        const int meshDim = mesh_->dim();

        const Index vertexEntities = mesh_->numCornerVertices();
        const Index edgeEntities = mesh_->numEdges();
        const Index faceEntities = (meshDim == 3) ? mesh_->numFaces() : mesh_->numElements();
        const Index volumeEntities = (meshDim == 3) ? mesh_->numElements() : 0;

        const Index vOffset = 0;
        const Index eOffset = vOffset + vertexEntities * layout.numVertexDofs;
        const Index fOffset = eOffset + edgeEntities * layout.numEdgeDofs;
        const Index cOffset = fOffset + faceEntities * layout.numFaceDofs;

        scalarNumDofs_ = cOffset + volumeEntities * layout.numVolumeDofs;
        numDofs_ = scalarNumDofs_ * vdim_;

        elemDofs_.assign(mesh_->numElements() * maxDofsPerElem_, InvalidIndex);
        bdrElemDofs_.assign(mesh_->numBdrElements() * maxDofsPerBdrElem_, InvalidIndex);

        const auto mapVertexDof = [&](Index vertexId, int k) -> Index {
            const Index cornerId = mesh_->vertexToCornerIndex(vertexId);
            if (cornerId == InvalidIndex) {
                return InvalidIndex;
            }
            return vOffset + cornerId * layout.numVertexDofs + k;
        };

        const auto mapEdgeDof = [&](Index edgeId, int k) -> Index {
            if (edgeId == InvalidIndex) {
                return InvalidIndex;
            }
            return eOffset + edgeId * layout.numEdgeDofs + k;
        };

        const auto mapFaceDof = [&](Index faceId, int k) -> Index {
            if (faceId == InvalidIndex) {
                return InvalidIndex;
            }
            return fOffset + faceId * layout.numFaceDofs + k;
        };

        for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
            const Element& elem = mesh_->element(elemIdx);
            const ReferenceElement* refElem = fec_->get(elem.geometry());
            if (!refElem) {
                continue;
            }

            const Index base = elemIdx * maxDofsPerElem_;
            int localDof = 0;

            const std::vector<Index> elemVertices = mesh_->getElementVertices(elemIdx);
            for (Index vId : elemVertices) {
                for (int k = 0; k < layout.numVertexDofs; ++k) {
                    elemDofs_[base + localDof++] = mapVertexDof(vId, k);
                }
            }

            const std::vector<Index> elemEdges = mesh_->getElementEdges(elemIdx);
            for (Index edgeId : elemEdges) {
                for (int k = 0; k < layout.numEdgeDofs; ++k) {
                    elemDofs_[base + localDof++] = mapEdgeDof(edgeId, k);
                }
            }

            if (layout.numFaceDofs > 0) {
                if (meshDim == 3) {
                    const std::vector<Index> elemFaces = mesh_->getElementFaces(elemIdx);
                    for (Index faceId : elemFaces) {
                        for (int k = 0; k < layout.numFaceDofs; ++k) {
                            elemDofs_[base + localDof++] = mapFaceDof(faceId, k);
                        }
                    }
                }
                else {
                    for (int k = 0; k < layout.numFaceDofs; ++k) {
                        elemDofs_[base + localDof++] = fOffset + elemIdx * layout.numFaceDofs + k;
                    }
                }
            }

            if (layout.numVolumeDofs > 0 && meshDim == 3) {
                for (int k = 0; k < layout.numVolumeDofs; ++k) {
                    elemDofs_[base + localDof++] = cOffset + elemIdx * layout.numVolumeDofs + k;
                }
            }

            if (localDof != refElem->numDofs()) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable local DOF count mismatch on volume element");
            }
        }

        for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
            const Element& bdrElem = mesh_->bdrElement(bdrIdx);
            const ReferenceElement* refElem = fec_->get(bdrElem.geometry());
            if (!refElem) {
                continue;
            }

            const Index base = bdrIdx * maxDofsPerBdrElem_;
            int localDof = 0;

            const int bdrCorners = bdrElem.numCorners();
            for (int i = 0; i < bdrCorners; ++i) {
                const Index vId = bdrElem.vertex(i);
                for (int k = 0; k < layout.numVertexDofs; ++k) {
                    bdrElemDofs_[base + localDof++] = mapVertexDof(vId, k);
                }
            }

            for (int localEdge = 0; localEdge < bdrElem.numEdges(); ++localEdge) {
                const auto [v0, v1] = bdrElem.edgeVertices(localEdge);
                const Index edgeId = mesh_->edgeIndex(v0, v1);
                for (int k = 0; k < layout.numEdgeDofs; ++k) {
                    bdrElemDofs_[base + localDof++] = mapEdgeDof(edgeId, k);
                }
            }

            if (layout.numFaceDofs > 0 && meshDim == 3 && geom::dim(bdrElem.geometry()) == 2) {
                const Index faceId = mesh_->getBoundaryFaceIndex(bdrIdx);
                for (int k = 0; k < layout.numFaceDofs; ++k) {
                    bdrElemDofs_[base + localDof++] = mapFaceDof(faceId, k);
                }
            }

            if (localDof != refElem->numDofs()) {
                MPFEM_THROW(Exception, "FESpace::buildDofTable local DOF count mismatch on boundary element");
            }
        }
    }

} // namespace mpfem

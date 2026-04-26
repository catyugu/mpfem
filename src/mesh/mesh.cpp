#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

    Mesh::Mesh(int dim, Index numVertices, Index numElements, Index numBdrElements)
        : dim_(dim)
    {
        if (numVertices > 0)
            reserveNodes(numVertices);
        if (numElements > 0)
            reserveElements(numElements);
        if (numBdrElements > 0)
            reserveBdrElements(numBdrElements);
    }

    void Mesh::setDim(int dim)
    {
        dim_ = dim;
        LOG_DEBUG << "Mesh dimension set to " << dim;
    }

    Index Mesh::addNode(Real x, Real y, Real z)
    {
        coords_.push_back(x);
        coords_.push_back(y);
        coords_.push_back(z);
        return static_cast<Index>(coords_.size() / dim_ - 1);
    }

    void Mesh::reserveNodes(Index n)
    {
        coords_.reserve(n * dim_);
    }

    Element Mesh::element(Index i) const
    {
        const Index start = elementOffsets_[i];
        const Index end = elementOffsets_[i + 1];
        const Index vertexCount = static_cast<Index>(geom::numVertices(elementGeoms_[i]));
        return Element {
            elementGeoms_[i],
            {&elementNodes_[start], static_cast<size_t>(vertexCount)},
            {&elementNodes_[start], static_cast<size_t>(end - start)},
            elementAttributes_[i],
            elementOrders_[i]};
    }

    Index Mesh::addElement(Geometry geom, std::span<const Index> nodes, Index attr, int order)
    {
        if (elementOffsets_.empty())
            elementOffsets_.push_back(0);
        elementGeoms_.push_back(geom);
        elementAttributes_.push_back(attr);
        elementOrders_.push_back(order);
        elementNodes_.insert(elementNodes_.end(), nodes.begin(), nodes.end());
        elementOffsets_.push_back(static_cast<Index>(elementNodes_.size()));
        return static_cast<Index>(elementGeoms_.size() - 1);
    }

    Index Mesh::addElement(Geometry geom, const std::vector<Index>& nodes, Index attr, int order)
    {
        return addElement(geom, std::span<const Index>(nodes), attr, order);
    }

    void Mesh::reserveElements(Index n)
    {
        elementGeoms_.reserve(n);
        elementAttributes_.reserve(n);
        elementOrders_.reserve(n);
        elementOffsets_.reserve(n + 1);
        elementNodes_.reserve(n * 8); // Estimate
    }

    Element Mesh::bdrElement(Index i) const
    {
        const Index start = bdrElementOffsets_[i];
        const Index end = bdrElementOffsets_[i + 1];
        const Index vertexCount = static_cast<Index>(geom::numVertices(bdrElementGeoms_[i]));
        return Element {
            bdrElementGeoms_[i],
            {&bdrElementNodes_[start], static_cast<size_t>(vertexCount)},
            {&bdrElementNodes_[start], static_cast<size_t>(end - start)},
            bdrElementAttributes_[i],
            bdrElementOrders_[i]};
    }

    Index Mesh::addBdrElement(Geometry geom, std::span<const Index> nodes, Index attr, int order)
    {
        if (bdrElementOffsets_.empty())
            bdrElementOffsets_.push_back(0);
        bdrElementGeoms_.push_back(geom);
        bdrElementAttributes_.push_back(attr);
        bdrElementOrders_.push_back(order);
        bdrElementNodes_.insert(bdrElementNodes_.end(), nodes.begin(), nodes.end());
        bdrElementOffsets_.push_back(static_cast<Index>(bdrElementNodes_.size()));
        return static_cast<Index>(bdrElementGeoms_.size() - 1);
    }

    Index Mesh::addBdrElement(Geometry geom, const std::vector<Index>& nodes, Index attr, int order)
    {
        return addBdrElement(geom, std::span<const Index>(nodes), attr, order);
    }

    void Mesh::reserveBdrElements(Index n)
    {
        bdrElementGeoms_.reserve(n);
        bdrElementAttributes_.reserve(n);
        bdrElementOrders_.reserve(n);
        bdrElementOffsets_.reserve(n + 1);
        bdrElementNodes_.reserve(n * 4); // Estimate
    }

    std::set<Index> Mesh::domainIds() const
    {
        std::set<Index> ids;
        for (Index i = 0; i < numElements(); ++i) {
            if (elementGeoms_[i] == Geometry::Tetrahedron || elementGeoms_[i] == Geometry::Cube) {
                ids.insert(elementAttributes_[i]);
            }
        }
        return ids;
    }

    std::set<Index> Mesh::boundaryIds() const
    {
        std::set<Index> ids;
        for (Index i = 0; i < numBdrElements(); ++i) {
            ids.insert(bdrElementAttributes_[i]);
        }
        return ids;
    }

    std::vector<Index> Mesh::elementsForDomain(Index domainId) const
    {
        std::vector<Index> result;
        for (Index i = 0; i < numElements(); ++i) {
            if ((elementGeoms_[i] == Geometry::Tetrahedron || elementGeoms_[i] == Geometry::Cube) && elementAttributes_[i] == domainId) {
                result.push_back(i);
            }
        }
        return result;
    }

    std::vector<Index> Mesh::bdrElementsForBoundary(Index boundaryId) const
    {
        std::vector<Index> result;
        for (Index i = 0; i < numBdrElements(); ++i) {
            if (bdrElementAttributes_[i] == boundaryId) {
                result.push_back(i);
            }
        }
        return result;
    }

    void Mesh::clear()
    {
        coords_.clear();
        elementGeoms_.clear();
        elementAttributes_.clear();
        elementOrders_.clear();
        elementOffsets_.clear();
        elementNodes_.clear();
        bdrElementGeoms_.clear();
        bdrElementAttributes_.clear();
        bdrElementOrders_.clear();
        bdrElementOffsets_.clear();
        bdrElementNodes_.clear();
        dim_ = 3;
        topologyBuilt_ = false;
        edgeInfoList_.clear();
        elemEdgeOffsets_.clear();
        elemEdgeData_.clear();
        faceOffsets_.clear();
        faceNodes_.clear();
        faceElem1_.clear();
        faceElem2_.clear();
        faceLocal1_.clear();
        faceLocal2_.clear();
        faceBoundary_.clear();
        elemFaceOffsets_.clear();
        elemFaceData_.clear();
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();
        bdrElementToFace_.clear();
        bdrIdExternalCache_.clear();
        sortedFaceKeys_.clear();
    }

    std::vector<Index> Mesh::getElementVertices(Index elemIdx) const
    {
        if (elemIdx >= numElements()) {
            return {};
        }

        const Element elem = element(elemIdx);
        const int corners = elem.numVertices();
        std::vector<Index> out;
        out.reserve(static_cast<size_t>(corners));
        for (int i = 0; i < corners; ++i) {
            out.push_back(elem.vertex(i));
        }
        return out;
    }

    std::span<const Index> Mesh::getElementEdges(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemEdgeOffsets_[elemIdx];
        const Index end = elemEdgeOffsets_[elemIdx + 1];
        return {elemEdgeData_.data() + start, elemEdgeData_.data() + end};
    }

    std::span<const Index> Mesh::getElementFaces(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemFaceOffsets_[elemIdx];
        const Index end = elemFaceOffsets_[elemIdx + 1];
        return {elemFaceData_.data() + start, elemFaceData_.data() + end};
    }

    Index Mesh::edgeIndex(Index a, Index b) const
    {
        // Binary search on sorted edgeInfoList_
        const Index lo = std::min(a, b);
        const Index hi = std::max(a, b);
        auto it = std::lower_bound(edgeInfoList_.begin(), edgeInfoList_.end(),
            EdgeInfo {lo, hi},
            [](const EdgeInfo& e1, const EdgeInfo& e2) {
                if (e1.v0 != e2.v0)
                    return e1.v0 < e2.v0;
                return e1.v1 < e2.v1;
            });
        if (it != edgeInfoList_.end() && it->v0 == lo && it->v1 == hi) {
            return static_cast<Index>(it - edgeInfoList_.begin());
        }
        return InvalidIndex;
    }

    // =============================================================================
    // Topology building
    // =============================================================================

    void Mesh::buildTopology()
    {
        if (topologyBuilt_)
            return;

        LOG_DEBUG << "Building mesh topology...";

        // Clear previous data
        edgeInfoList_.clear();
        elemEdgeOffsets_.clear();
        elemEdgeData_.clear();
        faceOffsets_.clear();
        faceNodes_.clear();
        faceElem1_.clear();
        faceElem2_.clear();
        faceLocal1_.clear();
        faceLocal2_.clear();
        faceBoundary_.clear();
        elemFaceOffsets_.clear();
        elemFaceData_.clear();
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();
        bdrElementToFace_.clear();
        sortedFaceKeys_.clear();

        // Build edge -> element mapping
        buildEdgeToElementMap();

        // Build face -> element mapping
        buildFaceToElementMap();

        // Build element -> face mapping
        buildElementToFaceMap();

        // Identify boundary faces
        identifyBoundaryFaces();

        // Build boundary element mapping
        buildBoundaryElementMapping();

        topologyBuilt_ = true;

        LOG_DEBUG << "Topology built: " << boundaryFaceIndices_.size() << " boundary faces, "
                  << interiorFaceIndices_.size() << " interior faces, "
                  << edgeInfoList_.size() << " edges, "
                  << bdrElementToFace_.size() << " boundary elements mapped";
    }

    void Mesh::buildEdgeToElementMap()
    {
        // First pass: collect all edges as (EdgeInfo, elemIdx, localEdge) tuples
        struct EdgeEntry {
            EdgeInfo edge;
            Index elemIdx;
            int localEdge;
        };
        std::vector<EdgeEntry> allEdges;

        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);
            const int nEdges = elem.numEdges();
            for (int localEdge = 0; localEdge < nEdges; ++localEdge) {
                const auto [v0, v1] = elem.edgeVertices(localEdge);
                allEdges.push_back(EdgeEntry {{std::min(v0, v1), std::max(v0, v1)}, elemIdx, localEdge});
            }
        }

        // Sort edges by (v0, v1)
        std::sort(allEdges.begin(), allEdges.end(), [](const EdgeEntry& a, const EdgeEntry& b) {
            if (a.edge.v0 != b.edge.v0)
                return a.edge.v0 < b.edge.v0;
            return a.edge.v1 < b.edge.v1;
        });

        // Deduplicate edges, building edgeInfoList_
        edgeInfoList_.clear();
        for (const auto& entry : allEdges) {
            if (edgeInfoList_.empty() || edgeInfoList_.back().v0 != entry.edge.v0 || edgeInfoList_.back().v1 != entry.edge.v1) {
                edgeInfoList_.push_back(entry.edge);
            }
        }

        // Build elemEdgeOffsets_
        elemEdgeOffsets_.clear();
        elemEdgeOffsets_.resize(numElements() + 1, 0);
        for (Index i = 0; i < numElements(); ++i) {
            elemEdgeOffsets_[i + 1] = elemEdgeOffsets_[i] + element(i).numEdges();
        }
        elemEdgeData_.resize(elemEdgeOffsets_.back());

        // Second pass: assign edge indices using binary search
        for (const auto& entry : allEdges) {
            auto it = std::lower_bound(edgeInfoList_.begin(), edgeInfoList_.end(), entry.edge,
                [](const EdgeInfo& e1, const EdgeInfo& e2) {
                    if (e1.v0 != e2.v0)
                        return e1.v0 < e2.v0;
                    return e1.v1 < e2.v1;
                });
            Index edgeIdx = static_cast<Index>(it - edgeInfoList_.begin());
            elemEdgeData_[elemEdgeOffsets_[entry.elemIdx] + entry.localEdge] = edgeIdx;
        }
    }

    void Mesh::buildFaceToElementMap()
    {
        // Stack-allocated face candidate to avoid heap allocation
        struct FaceEntry {
            FaceKey key; // Stack-allocated sorted node indices
            Index elemIdx;
            int localFace;
        };
        std::vector<FaceEntry> candidates;
        candidates.reserve(numElements() * 4); // Pre-reserve for typical mesh

        // Process each element
        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);

            for (int f = 0; f < elem.numFaces(); ++f) {
                auto faceVerts = elem.faceVertices(f);
                FaceEntry entry;
                entry.elemIdx = elemIdx;
                entry.localFace = f;
                // Sort nodes during insertion
                std::vector<Index> sorted(faceVerts.begin(), faceVerts.end());
                std::sort(sorted.begin(), sorted.end());
                entry.key.set(sorted);
                candidates.push_back(entry);
            }
        }

        // Sort by FaceKey for deduplication
        std::sort(candidates.begin(), candidates.end(), [](const FaceEntry& a, const FaceEntry& b) {
            return a.key < b.key;
        });

        // Deduplicate and build CSR face storage
        faceOffsets_.clear();
        faceOffsets_.push_back(0);
        faceElem1_.clear();
        faceElem2_.clear();
        faceLocal1_.clear();
        faceLocal2_.clear();
        faceBoundary_.clear();
        faceNodes_.clear();

        Index currentFaceIdx = 0;

        for (size_t i = 0; i < candidates.size();) {
            // Find all candidates with the same face
            size_t j = i;
            while (j < candidates.size() && candidates[j].key == candidates[i].key) {
                ++j;
            }

            // First occurrence = elem1, second = elem2
            const FaceEntry& first = candidates[i];
            Index e1 = first.elemIdx;
            Index e2 = InvalidIndex;
            int l1 = first.localFace;
            int l2 = -1;

            if (j - i > 1) {
                const FaceEntry& second = candidates[i + 1];
                e2 = second.elemIdx;
                l2 = second.localFace;
            }

            // Append face nodes to CSR storage
            for (int n = 0; n < first.key.count; ++n) {
                faceNodes_.push_back(first.key.nodes[n]);
            }
            faceOffsets_.push_back(static_cast<Index>(faceNodes_.size()));

            faceElem1_.push_back(e1);
            faceElem2_.push_back(e2);
            faceLocal1_.push_back(l1);
            faceLocal2_.push_back(l2);
            faceBoundary_.push_back(e2 == InvalidIndex ? 1 : 0);

            currentFaceIdx++;
            i = j;
        }

        // Build sortedFaceKeys_ for binary search in boundary element matching
        sortedFaceKeys_.clear();
        for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceElem1_.size()); ++faceIdx) {
            FaceKey key;
            std::span<const Index> nodes(&faceNodes_[faceOffsets_[faceIdx]],
                faceOffsets_[faceIdx + 1] - faceOffsets_[faceIdx]);
            key.set(nodes);
            sortedFaceKeys_.push_back({key, faceIdx});
        }
        // Sort by node vector for binary search
        std::sort(sortedFaceKeys_.begin(), sortedFaceKeys_.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    void Mesh::buildElementToFaceMap()
    {
        elemFaceOffsets_.clear();
        elemFaceOffsets_.resize(numElements() + 1, 0);

        for (Index i = 0; i < numElements(); ++i) {
            elemFaceOffsets_[i + 1] = elemFaceOffsets_[i] + element(i).numFaces();
        }
        elemFaceData_.assign(elemFaceOffsets_.back(), InvalidIndex);

        for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceElem1_.size()); ++faceIdx) {
            if (faceElem1_[faceIdx] != InvalidIndex) {
                elemFaceData_[elemFaceOffsets_[faceElem1_[faceIdx]] + faceLocal1_[faceIdx]] = faceIdx;
            }

            if (faceElem2_[faceIdx] != InvalidIndex) {
                elemFaceData_[elemFaceOffsets_[faceElem2_[faceIdx]] + faceLocal2_[faceIdx]] = faceIdx;
            }
        }
    }

    void Mesh::identifyBoundaryFaces()
    {
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();

        for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceElem1_.size()); ++faceIdx) {
            if (faceBoundary_[faceIdx] != 0) {
                boundaryFaceIndices_.push_back(faceIdx);
            }
            else {
                interiorFaceIndices_.push_back(faceIdx);
            }
        }
    }

    void Mesh::buildBoundaryElementMapping()
    {
        // Match boundary elements to topology faces
        Index externalCount = 0;
        Index internalCount = 0;
        bdrIdExternalCache_.clear();

        for (Index bdrIdx = 0; bdrIdx < numBdrElements(); ++bdrIdx) {
            const Element bdrElem = bdrElement(bdrIdx);
            Index bdrId = bdrElem.attribute;

            // Get sorted vertex key for boundary element - ONLY CORNER NODES
            // Build node array manually since FaceKey uses fixed array, not vector
            FaceKey key;
            int numVertices = bdrElem.numVertices();
            key.count = numVertices;
            for (int i = 0; i < numVertices; ++i) {
                key.nodes[i] = bdrElem.vertex(i);
            }
            // FaceKey's operator< handles lexicographic comparison (count first, then nodes)

            // Binary search in sortedFaceKeys_
            auto it = std::lower_bound(sortedFaceKeys_.begin(), sortedFaceKeys_.end(), key,
                [](const auto& pair, const FaceKey& k) { return pair.first < k; });
            if (it != sortedFaceKeys_.end() && it->first == key) {
                Index faceIdx = it->second;
                bdrElementToFace_[bdrIdx] = faceIdx;

                bool isExternal = faceBoundary_[faceIdx] != 0;

                if (isExternal) {
                    externalCount++;
                }
                else {
                    internalCount++;
                }

                // Cache boundary ID -> isExternal (first encounter sets the value)
                if (bdrIdExternalCache_.find(bdrId) == bdrIdExternalCache_.end()) {
                    bdrIdExternalCache_[bdrId] = isExternal;
                }
            }
        }

        LOG_INFO << "Boundary mapping: " << externalCount << " external, "
                 << internalCount << " internal (will skip in BC)";
    }

} // namespace mpfem
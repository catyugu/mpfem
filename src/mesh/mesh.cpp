#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

    // -----------------------------------------------------------------------------
    // Internal anonymous namespace - implementation details not exposed in header
    // -----------------------------------------------------------------------------
    namespace {
        /**
         * @brief Stack-allocated face key for topology building.
         *
         * Used during buildFaceTopology() to avoid millions of heap allocations
         * when using std::vector<Index>. Face node count is bounded (max 4 for quad),
         * so this fixed-size array is always sufficient.
         */
        struct FaceKey {
            static constexpr int MAX_FACE_NODES = 4;
            Index nodes[MAX_FACE_NODES];
            int count;

            FaceKey() : count(0) { }

            /// Initialize from sorted node span
            void set(std::span<const Index> sorted_nodes)
            {
                count = sorted_nodes.size();
                for (int i = 0; i < count; ++i)
                    nodes[i] = sorted_nodes[i];
            }

            /// Lexicographic comparison for sorting
            bool operator<(const FaceKey& o) const
            {
                if (count != o.count)
                    return count < o.count;
                for (int i = 0; i < count; ++i) {
                    if (nodes[i] != o.nodes[i])
                        return nodes[i] < o.nodes[i];
                }
                return false;
            }
            bool operator==(const FaceKey& o) const
            {
                if (count != o.count)
                    return false;
                for (int i = 0; i < count; ++i)
                    if (nodes[i] != o.nodes[i])
                        return false;
                return true;
            }
        };

        /// Edge entry for building edge topology
        struct EdgeEntry {
            Index v_min; // smaller vertex index (global positive direction)
            Index v_max; // larger vertex index
            Index elemIdx; // element index
            int localEdge; // local edge index in element
            int orientation; // 1 = same direction as global, -1 = reversed
        };

        /// Face entry for building face topology
        struct FaceEntry {
            FaceKey key;
            Index elemIdx;
            int localFace;
            int orientation; // 1 = elem1 (normal matches global), -1 = elem2 (reversed)
        };
    } // anonymous namespace

    // -----------------------------------------------------------------------------
    // Constructor and basic accessors
    // -----------------------------------------------------------------------------

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
        return Element {elementGeoms_[i],
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
        return Element {bdrElementGeoms_[i],
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

        // Edge topology
        edgeVertices_.clear();
        elemEdgeOffsets_.clear();
        elemEdgeData_.clear();
        elemEdgeOrientations_.clear();

        // Face topology
        faceOffsets_.clear();
        faceNodes_.clear();
        faceElem1_.clear();
        faceElem2_.clear();
        faceLocal1_.clear();
        faceLocal2_.clear();
        faceBoundary_.clear();
        elemFaceOffsets_.clear();
        elemFaceData_.clear();
        elemFaceOrientations_.clear();

        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();
        bdrElementToFace_.clear();
        bdrIdExternalCache_.clear();
    }

    // -----------------------------------------------------------------------------
    // Topology queries
    // -----------------------------------------------------------------------------

    std::span<const Index> Mesh::elementEdges(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemEdgeOffsets_[elemIdx];
        const Index end = elemEdgeOffsets_[elemIdx + 1];
        return {elemEdgeData_.data() + start, elemEdgeData_.data() + end};
    }

    std::span<const int> Mesh::elementEdgeOrientations(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemEdgeOffsets_[elemIdx];
        const Index end = elemEdgeOffsets_[elemIdx + 1];
        return {elemEdgeOrientations_.data() + start, elemEdgeOrientations_.data() + end};
    }

    std::span<const Index> Mesh::elementFaces(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemFaceOffsets_[elemIdx];
        const Index end = elemFaceOffsets_[elemIdx + 1];
        return {elemFaceData_.data() + start, elemFaceData_.data() + end};
    }

    std::span<const int> Mesh::elementFaceOrientations(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        const Index start = elemFaceOffsets_[elemIdx];
        const Index end = elemFaceOffsets_[elemIdx + 1];
        return {elemFaceOrientations_.data() + start, elemFaceOrientations_.data() + end};
    }

    Index Mesh::edgeIndex(Index a, Index b) const
    {
        const Index lo = std::min(a, b);
        const Index hi = std::max(a, b);

        Index count = edgeVertices_.size() / 2;
        Index left = 0, right = count; // [left, right) search range

        while (left < right) {
            Index mid = left + (right - left) / 2;
            Index midV0 = edgeVertices_[2 * mid];
            Index midV1 = edgeVertices_[2 * mid + 1];

            // Lexicographic comparison: first by v0, then by v1
            if (midV0 < lo || (midV0 == lo && midV1 < hi)) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }

        // Verify match at 'left'
        if (left < count) {
            Index v0 = edgeVertices_[2 * left];
            Index v1 = edgeVertices_[2 * left + 1];
            if (v0 == lo && v1 == hi) {
                return left;
            }
        }
        return InvalidIndex;
    }

    // -----------------------------------------------------------------------------
    // Topology building
    // -----------------------------------------------------------------------------

    void Mesh::buildTopology()
    {
        if (topologyBuilt_)
            return;

        LOG_DEBUG << "Building mesh topology...";

        buildEdgeTopology();
        buildFaceTopology();

        topologyBuilt_ = true;

        LOG_DEBUG << "Topology built: " << numBoundaryFaces() << " boundary faces, "
                  << numInteriorFaces() << " interior faces, " << numEdges() << " edges.";
    }

    void Mesh::buildEdgeTopology()
    {
        std::vector<EdgeEntry> allEdges;
        elemEdgeOffsets_.assign(numElements() + 1, 0);

        // First pass: collect all edges with their orientations
        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);
            const int nEdges = elem.numEdges();
            elemEdgeOffsets_[elemIdx + 1] = elemEdgeOffsets_[elemIdx] + nEdges;

            for (int localEdge = 0; localEdge < nEdges; ++localEdge) {
                const auto [v0, v1] = elem.edgeVertices(localEdge);
                // Global positive direction is v0 < v1
                // Orientation: 1 if element's local edge matches global direction, -1 otherwise
                int orientation = (v0 < v1) ? 1 : -1;
                allEdges.push_back(EdgeEntry {std::min(v0, v1), std::max(v0, v1), elemIdx, localEdge, orientation});
            }
        }

        // Sort edges by (v_min, v_max)
        std::sort(allEdges.begin(), allEdges.end(), [](const EdgeEntry& a, const EdgeEntry& b) {
            if (a.v_min != b.v_min)
                return a.v_min < b.v_min;
            return a.v_max < b.v_max;
        });

        // Build flat edgeVertices_ array and deduplicate
        edgeVertices_.clear();
        elemEdgeData_.resize(elemEdgeOffsets_.back());
        elemEdgeOrientations_.resize(elemEdgeOffsets_.back());

        Index currentGlobalEdgeIdx = InvalidIndex;
        Index last_v_min = InvalidIndex, last_v_max = InvalidIndex;

        for (const auto& entry : allEdges) {
            // New edge encountered
            if (entry.v_min != last_v_min || entry.v_max != last_v_max) {
                edgeVertices_.push_back(entry.v_min);
                edgeVertices_.push_back(entry.v_max);
                currentGlobalEdgeIdx++;
                last_v_min = entry.v_min;
                last_v_max = entry.v_max;
            }
            // Assign global edge index and orientation to element's local edge
            Index offset = elemEdgeOffsets_[entry.elemIdx] + entry.localEdge;
            elemEdgeData_[offset] = currentGlobalEdgeIdx;
            elemEdgeOrientations_[offset] = entry.orientation;
        }
    }

    void Mesh::buildFaceTopology()
    {
        std::vector<FaceEntry> candidates;
        elemFaceOffsets_.assign(numElements() + 1, 0);

        // First pass: collect all face candidates with sorted keys
        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);
            elemFaceOffsets_[elemIdx + 1] = elemFaceOffsets_[elemIdx] + elem.numFaces();

            for (int f = 0; f < elem.numFaces(); ++f) {
                auto faceVerts = elem.faceVertices(f);
                FaceEntry entry;
                entry.elemIdx = elemIdx;
                entry.localFace = f;

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

        // Initialize face storage
        faceOffsets_.assign(1, 0);
        faceNodes_.clear();
        faceElem1_.clear();
        faceElem2_.clear();
        faceLocal1_.clear();
        faceLocal2_.clear();
        faceBoundary_.clear();

        elemFaceData_.assign(elemFaceOffsets_.back(), InvalidIndex);
        elemFaceOrientations_.assign(elemFaceOffsets_.back(), 1); // Default: matches global

        // Local lookup table for boundary element matching (replaces sortedFaceKeys_ member)
        std::vector<std::pair<FaceKey, Index>> faceLookup;

        // Process deduplicated faces
        for (size_t i = 0; i < candidates.size();) {
            size_t j = i;
            while (j < candidates.size() && candidates[j].key == candidates[i].key) {
                ++j;
            }

            Index currentFaceIdx = static_cast<Index>(faceElem1_.size());
            const FaceEntry& first = candidates[i];

            // Append face nodes to CSR storage
            for (int n = 0; n < first.key.count; ++n) {
                faceNodes_.push_back(first.key.nodes[n]);
            }
            faceOffsets_.push_back(static_cast<Index>(faceNodes_.size()));

            // Determine adjacent elements
            Index e1 = first.elemIdx;
            int l1 = first.localFace;
            Index e2 = InvalidIndex;
            int l2 = -1;

            if (j - i > 1) {
                const FaceEntry& second = candidates[i + 1];
                e2 = second.elemIdx;
                l2 = second.localFace;
            }

            faceElem1_.push_back(e1);
            faceElem2_.push_back(e2);
            faceLocal1_.push_back(l1);
            faceLocal2_.push_back(l2);
            faceBoundary_.push_back(e2 == InvalidIndex ? 1 : 0);

            // Set element-to-face mapping
            // For elem1: orientation = 1 (face normal matches global)
            elemFaceData_[elemFaceOffsets_[e1] + l1] = currentFaceIdx;
            elemFaceOrientations_[elemFaceOffsets_[e1] + l1] = 1;

            // For elem2: orientation = -1 (face normal opposite to global)
            if (e2 != InvalidIndex) {
                elemFaceData_[elemFaceOffsets_[e2] + l2] = currentFaceIdx;
                elemFaceOrientations_[elemFaceOffsets_[e2] + l2] = -1;
            }

            // Track boundary/interior faces
            if (e2 == InvalidIndex) {
                boundaryFaceIndices_.push_back(currentFaceIdx);
            }
            else {
                interiorFaceIndices_.push_back(currentFaceIdx);
            }

            // Add to local lookup table for boundary matching
            faceLookup.push_back({first.key, currentFaceIdx});

            i = j;
        }

        // --- Boundary element mapping (local, replaces member variable sortedFaceKeys_) ---
        bdrIdExternalCache_.clear();

        for (Index bdrIdx = 0; bdrIdx < numBdrElements(); ++bdrIdx) {
            const Element bdrElem = bdrElement(bdrIdx);
            FaceKey key;
            key.count = bdrElem.numVertices();
            for (int i = 0; i < key.count; ++i) {
                key.nodes[i] = bdrElem.vertex(i);
            }
            // Sort key for binary search (FaceKey's operator< handles lexicographic comparison)

            auto it = std::lower_bound(faceLookup.begin(), faceLookup.end(), key,
                [](const auto& pair, const FaceKey& k) { return pair.first < k; });

            if (it != faceLookup.end() && it->first == key) {
                Index faceIdx = it->second;
                bdrElementToFace_[bdrIdx] = faceIdx;

                // Cache boundary ID -> isExternal (first encounter sets the value)
                if (bdrIdExternalCache_.find(bdrElem.attribute) == bdrIdExternalCache_.end()) {
                    bdrIdExternalCache_[bdrElem.attribute] = (faceBoundary_[faceIdx] != 0);
                }
            }
        }
    }

} // namespace mpfem
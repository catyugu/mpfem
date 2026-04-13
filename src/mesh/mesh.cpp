#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

    std::uint64_t Mesh::edgeKey(Index a, Index b)
    {
        const Index lo = std::min(a, b);
        const Index hi = std::max(a, b);
        return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    }

    Mesh::Mesh(int dim, Index numVertices, Index numElements, Index numBdrElements)
        : dim_(dim)
    {
        if (numVertices > 0)
            vertices_.reserve(numVertices);
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

    void Mesh::addVertex(const Vector3& v)
    {
        vertices_.push_back(v);
    }

    void Mesh::addVertex(Vector3&& v)
    {
        vertices_.push_back(std::move(v));
    }

    Index Mesh::addVertex(Real x, Real y, Real z)
    {
        vertices_.emplace_back(x, y, z);
        return static_cast<Index>(vertices_.size() - 1);
    }

    void Mesh::reserveVertices(Index n)
    {
        vertices_.reserve(n);
    }

    Element Mesh::element(Index i) const
    {
        const Index start = elementOffsets_[i];
        const Index end = elementOffsets_[i + 1];
        return Element {
            elementGeoms_[i],
            {&elementVertices_[start], static_cast<size_t>(end - start)},
            elementAttributes_[i],
            elementOrders_[i]};
    }

    Index Mesh::addElement(Geometry geom, std::span<const Index> vertices, Index attr, int order)
    {
        if (elementOffsets_.empty())
            elementOffsets_.push_back(0);
        elementGeoms_.push_back(geom);
        elementAttributes_.push_back(attr);
        elementOrders_.push_back(order);
        elementVertices_.insert(elementVertices_.end(), vertices.begin(), vertices.end());
        elementOffsets_.push_back(static_cast<Index>(elementVertices_.size()));
        return static_cast<Index>(elementGeoms_.size() - 1);
    }

    Index Mesh::addElement(Geometry geom, const std::vector<Index>& vertices, Index attr, int order)
    {
        return addElement(geom, std::span<const Index>(vertices), attr, order);
    }

    void Mesh::reserveElements(Index n)
    {
        elementGeoms_.reserve(n);
        elementAttributes_.reserve(n);
        elementOrders_.reserve(n);
        elementOffsets_.reserve(n + 1);
        elementVertices_.reserve(n * 8); // Estimate
    }

    Element Mesh::bdrElement(Index i) const
    {
        const Index start = bdrElementOffsets_[i];
        const Index end = bdrElementOffsets_[i + 1];
        return Element {
            bdrElementGeoms_[i],
            {&bdrElementVertices_[start], static_cast<size_t>(end - start)},
            bdrElementAttributes_[i],
            bdrElementOrders_[i]};
    }

    Index Mesh::addBdrElement(Geometry geom, std::span<const Index> vertices, Index attr, int order)
    {
        if (bdrElementOffsets_.empty())
            bdrElementOffsets_.push_back(0);
        bdrElementGeoms_.push_back(geom);
        bdrElementAttributes_.push_back(attr);
        bdrElementOrders_.push_back(order);
        bdrElementVertices_.insert(bdrElementVertices_.end(), vertices.begin(), vertices.end());
        bdrElementOffsets_.push_back(static_cast<Index>(bdrElementVertices_.size()));
        return static_cast<Index>(bdrElementGeoms_.size() - 1);
    }

    Index Mesh::addBdrElement(Geometry geom, const std::vector<Index>& vertices, Index attr, int order)
    {
        return addBdrElement(geom, std::span<const Index>(vertices), attr, order);
    }

    void Mesh::reserveBdrElements(Index n)
    {
        bdrElementGeoms_.reserve(n);
        bdrElementAttributes_.reserve(n);
        bdrElementOrders_.reserve(n);
        bdrElementOffsets_.reserve(n + 1);
        bdrElementVertices_.reserve(n * 4); // Estimate
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
        vertices_.clear();
        elementGeoms_.clear();
        elementAttributes_.clear();
        elementOrders_.clear();
        elementOffsets_.clear();
        elementVertices_.clear();
        bdrElementGeoms_.clear();
        bdrElementAttributes_.clear();
        bdrElementOrders_.clear();
        bdrElementOffsets_.clear();
        bdrElementVertices_.clear();
        dim_ = 3;
        topologyBuilt_ = false;
        edgeInfoList_.clear();
        edgeKeyToIndex_.clear();
        elemEdgeOffsets_.clear();
        elemEdgeData_.clear();
        faceInfoList_.clear();
        faceKeyToIndex_.clear();
        elemFaceOffsets_.clear();
        elemFaceData_.clear();
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();
        bdrElementToFace_.clear();
        bdrIdExternalCache_.clear();
        cornerVertexIndices_.clear();
        cornerVertexMap_.clear();
    }

    std::vector<Index> Mesh::getElementVertices(Index elemIdx) const
    {
        if (elemIdx >= numElements()) {
            return {};
        }

        const Element elem = element(elemIdx);
        const int corners = elem.numCorners();
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
        return {&elemEdgeData_[elemEdgeOffsets_[elemIdx]],
            &elemEdgeData_[elemEdgeOffsets_[elemIdx + 1]]};
    }

    std::span<const Index> Mesh::getElementFaces(Index elemIdx) const
    {
        if (!topologyBuilt_ || elemIdx >= numElements()) {
            return {};
        }
        return {&elemFaceData_[elemFaceOffsets_[elemIdx]],
            &elemFaceData_[elemFaceOffsets_[elemIdx + 1]]};
    }

    Index Mesh::edgeIndex(Index a, Index b) const
    {
        const auto it = edgeKeyToIndex_.find(edgeKey(a, b));
        return (it == edgeKeyToIndex_.end()) ? InvalidIndex : it->second;
    }

    std::pair<Vector3, Vector3> Mesh::getBoundingBox() const
    {
        if (vertices_.empty()) {
            return {Vector3::Zero(), Vector3::Zero()};
        }

        Vector3 minCoord = vertices_[0];
        Vector3 maxCoord = minCoord;

        for (const auto& v : vertices_) {
            minCoord = minCoord.cwiseMin(v);
            maxCoord = maxCoord.cwiseMax(v);
        }

        return {minCoord, maxCoord};
    }

    // =============================================================================
    // Corner vertices (topological vertices)
    // =============================================================================

    Index Mesh::numCornerVertices() const
    {
        return static_cast<Index>(cornerVertexIndices_.size());
    }

    const std::vector<Index>& Mesh::cornerVertexIndices() const
    {
        return cornerVertexIndices_;
    }

    Index Mesh::vertexToCornerIndex(Index vertexIdx) const
    {
        if (vertexIdx >= static_cast<Index>(cornerVertexMap_.size())) {
            return InvalidIndex;
        }
        return cornerVertexMap_[vertexIdx];
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
        edgeKeyToIndex_.clear();
        elemEdgeOffsets_.clear();
        elemEdgeData_.clear();
        faceInfoList_.clear();
        faceKeyToIndex_.clear();
        elemFaceOffsets_.clear();
        elemFaceData_.clear();
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();
        bdrElementToFace_.clear();

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

        // Build corner vertex map (eagerly, for high-order meshes)
        // Collect all corner vertices from all volume elements
        std::set<Index> cornerSet;
        for (Index i = 0; i < numElements(); ++i) {
            const Element e = element(i);
            int nc = e.numCorners();
            for (int j = 0; j < nc; ++j) {
                cornerSet.insert(e.vertex(j));
            }
        }

        // Build sorted list of corner vertex indices
        cornerVertexIndices_.clear();
        cornerVertexIndices_.reserve(cornerSet.size());
        for (Index v : cornerSet) {
            cornerVertexIndices_.push_back(v);
        }
        std::sort(cornerVertexIndices_.begin(), cornerVertexIndices_.end());

        // Build reverse mapping: full vertex index -> corner vertex index (or InvalidIndex)
        cornerVertexMap_.clear();
        cornerVertexMap_.resize(numVertices(), InvalidIndex);
        for (Index i = 0; i < static_cast<Index>(cornerVertexIndices_.size()); ++i) {
            cornerVertexMap_[cornerVertexIndices_[i]] = i;
        }

        topologyBuilt_ = true;

        LOG_DEBUG << "Topology built: " << boundaryFaceIndices_.size() << " boundary faces, "
                  << interiorFaceIndices_.size() << " interior faces, "
                  << edgeInfoList_.size() << " edges, "
                  << bdrElementToFace_.size() << " boundary elements mapped, "
                  << cornerVertexIndices_.size() << " corner vertices";
    }

    void Mesh::buildEdgeToElementMap()
    {
        elemEdgeOffsets_.clear();
        elemEdgeOffsets_.resize(numElements() + 1, 0);

        for (Index i = 0; i < numElements(); ++i) {
            elemEdgeOffsets_[i + 1] = elemEdgeOffsets_[i] + element(i).numEdges();
        }
        elemEdgeData_.resize(elemEdgeOffsets_.back());

        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);
            const int nEdges = elem.numEdges();
            const Index base = elemEdgeOffsets_[elemIdx];

            for (int localEdge = 0; localEdge < nEdges; ++localEdge) {
                const auto [v0, v1] = elem.edgeVertices(localEdge);
                const std::uint64_t key = edgeKey(v0, v1);

                auto it = edgeKeyToIndex_.find(key);
                Index edgeIdx = InvalidIndex;
                if (it == edgeKeyToIndex_.end()) {
                    edgeIdx = static_cast<Index>(edgeInfoList_.size());
                    edgeInfoList_.push_back(EdgeInfo {std::min(v0, v1), std::max(v0, v1)});
                    edgeKeyToIndex_[key] = edgeIdx;
                }
                else {
                    edgeIdx = it->second;
                }

                elemEdgeData_[base + localEdge] = edgeIdx;
            }
        }
    }

    void Mesh::buildFaceToElementMap()
    {
        // Temporary map for building
        std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> faceMap;

        // Process each element
        for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
            const Element elem = element(elemIdx);

            // Get all faces of this element
            for (int f = 0; f < elem.numFaces(); ++f) {
                auto faceVerts = elem.faceVertices(f);

                // Create a sorted key for the face
                FaceKey key;
                key.count = 0;
                for (Index v : faceVerts) {
                    key.nodes[key.count++] = v;
                }
                std::sort(key.nodes, key.nodes + key.count);

                // Check if face already exists
                auto it = faceMap.find(key);
                if (it == faceMap.end()) {
                    // New face
                    FaceInfo info;
                    info.elem1 = elemIdx;
                    info.elem2 = InvalidIndex;
                    info.localFace1 = f;
                    info.localFace2 = -1;
                    info.isBoundary = true;
                    info.vertices = faceVerts;

                    faceMap[key] = std::move(info);
                }
                else {
                    // Face already exists - this is an interior face
                    it->second.elem2 = elemIdx;
                    it->second.localFace2 = f;
                    it->second.isBoundary = false;
                }
            }
        }

        // Transfer to vector for O(1) access
        faceInfoList_.reserve(faceMap.size());
        Index faceIdx = 0;
        for (auto& [key, info] : faceMap) {
            faceInfoList_.push_back(std::move(info));
            faceKeyToIndex_[key] = faceIdx++;
        }
    }

    void Mesh::buildElementToFaceMap()
    {
        elemFaceOffsets_.clear();
        elemFaceOffsets_.resize(numElements() + 1, 0);

        for (Index i = 0; i < numElements(); ++i) {
            elemFaceOffsets_[i + 1] = elemFaceOffsets_[i] + element(i).numFaces();
        }
        elemFaceData_.assign(elemFaceOffsets_.back(), InvalidIndex);

        for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceInfoList_.size()); ++faceIdx) {
            const auto& info = faceInfoList_[faceIdx];

            if (info.elem1 != InvalidIndex) {
                elemFaceData_[elemFaceOffsets_[info.elem1] + info.localFace1] = faceIdx;
            }

            if (info.elem2 != InvalidIndex) {
                elemFaceData_[elemFaceOffsets_[info.elem2] + info.localFace2] = faceIdx;
            }
        }
    }

    void Mesh::identifyBoundaryFaces()
    {
        boundaryFaceIndices_.clear();
        interiorFaceIndices_.clear();

        for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceInfoList_.size()); ++faceIdx) {
            if (faceInfoList_[faceIdx].isBoundary) {
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
        // A boundary element should match a boundary face by its CORNER vertices only
        // (not including edge midpoints for quadratic elements)

        Index externalCount = 0;
        Index internalCount = 0;
        bdrIdExternalCache_.clear();

        for (Index bdrIdx = 0; bdrIdx < numBdrElements(); ++bdrIdx) {
            const Element bdrElem = bdrElement(bdrIdx);
            Index bdrId = bdrElem.attribute;

            // Get sorted vertex key for boundary element - ONLY CORNER NODES
            FaceKey key;
            int numCorners = bdrElem.numCorners();
            key.count = numCorners;
            for (int i = 0; i < numCorners; ++i) {
                key.nodes[i] = bdrElem.vertex(i);
            }
            std::sort(key.nodes, key.nodes + key.count);

            // Find matching face
            auto it = faceKeyToIndex_.find(key);
            if (it != faceKeyToIndex_.end()) {
                Index faceIdx = it->second;
                bdrElementToFace_[bdrIdx] = faceIdx;

                bool isExternal = faceInfoList_[faceIdx].isBoundary;

                // Count external vs internal boundaries
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

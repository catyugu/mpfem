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
    : dim_(dim) {
    // Use reserve instead of resize to allow automatic expansion
    if (numVertices > 0) vertices_.reserve(numVertices);
    if (numElements > 0) elements_.reserve(numElements);
    if (numBdrElements > 0) bdrElements_.reserve(numBdrElements);
}

void Mesh::setDim(int dim) {
    dim_ = dim;
    LOG_DEBUG << "Mesh dimension set to " << dim;
}

void Mesh::addVertex(const Vertex& v) {
    vertices_.push_back(v);
}

void Mesh::addVertex(Vertex&& v) {
    vertices_.push_back(std::move(v));
}

Index Mesh::addVertex(Real x, Real y, Real z) {
    vertices_.emplace_back(x, y, z, dim_);
    return static_cast<Index>(vertices_.size() - 1);
}

void Mesh::reserveVertices(Index n) {
    vertices_.reserve(n);
}

void Mesh::addElement(const Element& e) {
    elements_.push_back(e);
}

void Mesh::addElement(Element&& e) {
    elements_.push_back(std::move(e));
}

Index Mesh::addElement(Geometry geom, std::span<const Index> vertices, Index attr, int order) {
    elements_.emplace_back(geom, vertices, attr, order);
    return static_cast<Index>(elements_.size() - 1);
}

Index Mesh::addElement(Geometry geom, const std::vector<Index>& vertices, Index attr, int order) {
    elements_.emplace_back(geom, vertices, attr, order);
    return static_cast<Index>(elements_.size() - 1);
}

void Mesh::reserveElements(Index n) {
    elements_.reserve(n);
}

void Mesh::addBdrElement(const Element& e) {
    bdrElements_.push_back(e);
}

void Mesh::addBdrElement(Element&& e) {
    bdrElements_.push_back(std::move(e));
}

Index Mesh::addBdrElement(Geometry geom, std::span<const Index> vertices, Index attr, int order) {
    bdrElements_.emplace_back(geom, vertices, attr, order);
    return static_cast<Index>(bdrElements_.size() - 1);
}

Index Mesh::addBdrElement(Geometry geom, const std::vector<Index>& vertices, Index attr, int order) {
    bdrElements_.emplace_back(geom, vertices, attr, order);
    return static_cast<Index>(bdrElements_.size() - 1);
}

void Mesh::reserveBdrElements(Index n) {
    bdrElements_.reserve(n);
}

std::set<Index> Mesh::domainIds() const {
    std::set<Index> ids;
    for (const auto& e : elements_) {
        // Only count volume elements (tetrahedra, hexahedra) as domains
        if (e.geometry() == Geometry::Tetrahedron || 
            e.geometry() == Geometry::Cube) {
            ids.insert(e.attribute());
        }
    }
    return ids;
}

std::set<Index> Mesh::boundaryIds() const {
    std::set<Index> ids;
    for (const auto& e : bdrElements_) {
        ids.insert(e.attribute());
    }
    return ids;
}

std::vector<Index> Mesh::elementsForDomain(Index domainId) const {
    std::vector<Index> result;
    for (Index i = 0; i < static_cast<Index>(elements_.size()); ++i) {
        const auto& e = elements_[i];
        // Only count volume elements
        if ((e.geometry() == Geometry::Tetrahedron || 
             e.geometry() == Geometry::Cube) &&
            e.attribute() == domainId) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<Index> Mesh::bdrElementsForBoundary(Index boundaryId) const {
    std::vector<Index> result;
    for (Index i = 0; i < static_cast<Index>(bdrElements_.size()); ++i) {
        if (bdrElements_[i].attribute() == boundaryId) {
            result.push_back(i);
        }
    }
    return result;
}

void Mesh::clear() {
    vertices_.clear();
    elements_.clear();
    bdrElements_.clear();
    dim_ = 3;
    topologyBuilt_ = false;
    edgeInfoList_.clear();
    edgeKeyToIndex_.clear();
    elementToEdge_.clear();
    faceInfoList_.clear();
    faceKeyToIndex_.clear();
    elementToFace_.clear();
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

    const Element& elem = element(elemIdx);
    const int corners = elem.numCorners();
    std::vector<Index> out;
    out.reserve(static_cast<size_t>(corners));
    for (int i = 0; i < corners; ++i) {
        out.push_back(elem.vertex(i));
    }
    return out;
}

std::vector<Index> Mesh::getElementEdges(Index elemIdx) const
{
    if (elemIdx >= static_cast<Index>(elementToEdge_.size())) {
        return {};
    }

    std::vector<std::pair<int, Index>> localToGlobal = elementToEdge_[elemIdx];
    std::sort(localToGlobal.begin(), localToGlobal.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<Index> out;
    out.reserve(localToGlobal.size());
    for (const auto& [localEdge, globalEdge] : localToGlobal) {
        (void)localEdge;
        out.push_back(globalEdge);
    }
    return out;
}

std::vector<Index> Mesh::getElementFaces(Index elemIdx) const
{
    if (elemIdx >= static_cast<Index>(elementToFace_.size())) {
        return {};
    }

    std::vector<std::pair<int, Index>> localToGlobal = elementToFace_[elemIdx];
    std::sort(localToGlobal.begin(), localToGlobal.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<Index> out;
    out.reserve(localToGlobal.size());
    for (const auto& [localFace, globalFace] : localToGlobal) {
        (void)localFace;
        out.push_back(globalFace);
    }
    return out;
}

Index Mesh::edgeIndex(Index a, Index b) const
{
    const auto it = edgeKeyToIndex_.find(edgeKey(a, b));
    return (it == edgeKeyToIndex_.end()) ? InvalidIndex : it->second;
}

std::pair<Vector3, Vector3> Mesh::getBoundingBox() const {
    if (vertices_.empty()) {
        return {Vector3::Zero(), Vector3::Zero()};
    }
    
    Vector3 minCoord = vertices_[0].toVector();
    Vector3 maxCoord = minCoord;
    
    for (const auto& v : vertices_) {
        Vector3 p = v.toVector();
        minCoord = minCoord.cwiseMin(p);
        maxCoord = maxCoord.cwiseMax(p);
    }
    
    return {minCoord, maxCoord};
}

// =============================================================================
// Corner vertices (topological vertices)
// =============================================================================

Index Mesh::numCornerVertices() const {
    return static_cast<Index>(cornerVertexIndices_.size());
}

const std::vector<Index>& Mesh::cornerVertexIndices() const {
    return cornerVertexIndices_;
}

Index Mesh::vertexToCornerIndex(Index vertexIdx) const {
    if (vertexIdx >= static_cast<Index>(cornerVertexMap_.size())) {
        return InvalidIndex;
    }
    return cornerVertexMap_[vertexIdx];
}

// =============================================================================
// Topology building
// =============================================================================

void Mesh::buildTopology() {
    if (topologyBuilt_) return;
    
    LOG_DEBUG << "Building mesh topology...";
    
    // Clear previous data
    edgeInfoList_.clear();
    edgeKeyToIndex_.clear();
    elementToEdge_.clear();
    faceInfoList_.clear();
    faceKeyToIndex_.clear();
    elementToFace_.clear();
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
    for (const auto& e : elements_) {
        int nc = e.numCorners();
        for (int i = 0; i < nc; ++i) {
            cornerSet.insert(e.vertex(i));
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

void Mesh::buildEdgeToElementMap() {
    elementToEdge_.clear();
    elementToEdge_.resize(numElements());

    for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
        const Element& elem = element(elemIdx);
        const int nEdges = elem.numEdges();

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

            elementToEdge_[elemIdx].push_back({localEdge, edgeIdx});
        }
    }
}

void Mesh::buildFaceToElementMap() {
    // Temporary map for building
    std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> faceMap;
    
    // Process each element
    for (Index elemIdx = 0; elemIdx < numElements(); ++elemIdx) {
        const Element& elem = element(elemIdx);
        
        // Get all faces of this element
        for (int f = 0; f < elem.numFaces(); ++f) {
            auto faceVerts = elem.faceVertices(f);
            
            // Create a sorted key for the face
            FaceKey key;
            key.reserve(faceVerts.size());
            for (Index v : faceVerts) {
                key.push_back(v);
            }
            std::sort(key.begin(), key.end());
            
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
            } else {
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

void Mesh::buildElementToFaceMap() {
    elementToFace_.clear();
    elementToFace_.resize(numElements());
    
    // Build element to face mapping using the face index
    for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceInfoList_.size()); ++faceIdx) {
        const auto& info = faceInfoList_[faceIdx];
        
        // Add face to element 1
        if (info.elem1 != InvalidIndex && info.elem1 < static_cast<Index>(elementToFace_.size())) {
            elementToFace_[info.elem1].push_back({info.localFace1, faceIdx});
        }
        
        // Add face to element 2 (if exists)
        if (info.elem2 != InvalidIndex && info.elem2 < static_cast<Index>(elementToFace_.size())) {
            elementToFace_[info.elem2].push_back({info.localFace2, faceIdx});
        }
    }
}

void Mesh::identifyBoundaryFaces() {
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    
    for (Index faceIdx = 0; faceIdx < static_cast<Index>(faceInfoList_.size()); ++faceIdx) {
        if (faceInfoList_[faceIdx].isBoundary) {
            boundaryFaceIndices_.push_back(faceIdx);
        } else {
            interiorFaceIndices_.push_back(faceIdx);
        }
    }
}

void Mesh::buildBoundaryElementMapping() {
    // Match boundary elements to topology faces
    // A boundary element should match a boundary face by its CORNER vertices only
    // (not including edge midpoints for quadratic elements)
    
    Index externalCount = 0;
    Index internalCount = 0;
    bdrIdExternalCache_.clear();
    
    for (Index bdrIdx = 0; bdrIdx < numBdrElements(); ++bdrIdx) {
        const Element& bdrElem = bdrElement(bdrIdx);
        Index bdrId = bdrElem.attribute();
        
        // Get sorted vertex key for boundary element - ONLY CORNER NODES
        FaceKey key;
        int numCorners = bdrElem.numCorners();
        key.reserve(numCorners);
        for (int i = 0; i < numCorners; ++i) {
            key.push_back(bdrElem.vertex(i));
        }
        std::sort(key.begin(), key.end());
        
        // Find matching face
        auto it = faceKeyToIndex_.find(key);
        if (it != faceKeyToIndex_.end()) {
            Index faceIdx = it->second;
            bdrElementToFace_[bdrIdx] = faceIdx;
            
            bool isExternal = faceInfoList_[faceIdx].isBoundary;
            
            // Count external vs internal boundaries
            if (isExternal) {
                externalCount++;
            } else {
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

}  // namespace mpfem

#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

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
    faceInfoList_.clear();
    faceKeyToIndex_.clear();
    elementToFace_.clear();
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    bdrElementToFace_.clear();
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
// Topology building
// =============================================================================

void Mesh::buildTopology() {
    if (topologyBuilt_) return;
    
    LOG_DEBUG << "Building mesh topology...";
    
    // Clear previous data
    faceInfoList_.clear();
    faceKeyToIndex_.clear();
    elementToFace_.clear();
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    bdrElementToFace_.clear();
    
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
              << bdrElementToFace_.size() << " boundary elements mapped";
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
    // A boundary element should match a boundary face by its vertices
    
    for (Index bdrIdx = 0; bdrIdx < numBdrElements(); ++bdrIdx) {
        const Element& bdrElem = bdrElement(bdrIdx);
        
        // Get sorted vertex key for boundary element
        FaceKey key;
        const auto& verts = bdrElem.vertices();
        key.reserve(verts.size());
        for (Index v : verts) {
            key.push_back(v);
        }
        std::sort(key.begin(), key.end());
        
        // Find matching face
        auto it = faceKeyToIndex_.find(key);
        if (it != faceKeyToIndex_.end()) {
            Index faceIdx = it->second;
            bdrElementToFace_[bdrIdx] = faceIdx;
        }
    }
}

}  // namespace mpfem
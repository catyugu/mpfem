#include "mesh/mesh_topology.hpp"
#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

MeshTopology::MeshTopology(const Mesh* mesh)
    : mesh_(mesh) {
    if (mesh_) {
        build();
    }
}

void MeshTopology::setMesh(const Mesh* mesh) {
    mesh_ = mesh;
    if (mesh_) {
        build();
    }
}

void MeshTopology::build() {
    if (!mesh_) return;
    
    LOG_INFO << "Building mesh topology...";
    
    // Clear previous data
    faceInfoList_.clear();
    faceKeyToIndex_.clear();
    elementToFace_.clear();
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    bdrElementToFace_.clear();
    faceToBdrElement_.clear();
    
    // Build face -> element mapping
    buildFaceToElementMap();
    
    // Build element -> face mapping
    buildElementToFaceMap();
    
    // Identify boundary faces
    identifyBoundaryFaces();
    
    // Build boundary element mapping
    buildBoundaryElementMapping();
    
    LOG_INFO << "Topology built: " << boundaryFaceIndices_.size() << " boundary faces, "
             << interiorFaceIndices_.size() << " interior faces, "
             << bdrElementToFace_.size() << " boundary elements mapped";
}

void MeshTopology::buildFaceToElementMap() {
    // Temporary map for building
    std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> faceMap;
    
    // Process each element
    for (Index elemIdx = 0; elemIdx < mesh_->numElements(); ++elemIdx) {
        const Element& elem = mesh_->element(elemIdx);
        
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

void MeshTopology::buildElementToFaceMap() {
    elementToFace_.clear();
    elementToFace_.resize(mesh_->numElements());
    
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

void MeshTopology::identifyBoundaryFaces() {
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

void MeshTopology::buildBoundaryElementMapping() {
    // Match boundary elements to topology faces
    // A boundary element should match a boundary face by its vertices
    
    for (Index bdrIdx = 0; bdrIdx < mesh_->numBdrElements(); ++bdrIdx) {
        const Element& bdrElem = mesh_->bdrElement(bdrIdx);
        
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
            faceToBdrElement_[faceIdx] = bdrIdx;
        }
    }
}

std::vector<Index> MeshTopology::getBoundaryElementsForBoundary(Index boundaryId) const {
    std::vector<Index> result;
    
    for (Index i = 0; i < mesh_->numBdrElements(); ++i) {
        if (mesh_->bdrElement(i).attribute() == boundaryId) {
            result.push_back(i);
        }
    }
    
    return result;
}

}  // namespace mpfem

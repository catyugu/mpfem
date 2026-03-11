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
    
    LOG_INFO("Building mesh topology...");
    
    // Clear previous data
    faceToElement_.clear();
    faceVertices_.clear();
    faceToIndex_.clear();
    elementToFace_.clear();
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    
    // Build face -> element mapping
    buildFaceToElementMap();
    
    // Build element -> face mapping
    buildElementToFaceMap();
    
    // Identify boundary faces
    identifyBoundaryFaces();
    
    LOG_INFO("Topology built: " << boundaryFaceIndices_.size() << " boundary faces, "
             << interiorFaceIndices_.size() << " interior faces");
}

void MeshTopology::buildFaceToElementMap() {
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
            auto it = faceToElement_.find(key);
            if (it == faceToElement_.end()) {
                // New face
                FaceInfo info;
                info.elem1 = elemIdx;
                info.elem2 = InvalidIndex;
                info.localFace1 = f;
                info.localFace2 = -1;
                info.isBoundary = true;
                
                faceToElement_[key] = info;
                faceVertices_[key] = std::move(faceVerts);
            } else {
                // Face already exists - this is an interior face
                it->second.elem2 = elemIdx;
                it->second.localFace2 = f;
                it->second.isBoundary = false;
            }
        }
    }
}

void MeshTopology::buildElementToFaceMap() {
    elementToFace_.clear();
    elementToFace_.resize(mesh_->numElements());
    
    // First assign face indices
    Index faceIdx = 0;
    for (const auto& [key, info] : faceToElement_) {
        faceToIndex_[key] = faceIdx++;
    }
    
    // Now build element to face mapping
    for (const auto& [key, info] : faceToElement_) {
        Index idx = faceToIndex_[key];
        
        // Add face to element 1
        if (info.elem1 >= 0 && info.elem1 < static_cast<Index>(elementToFace_.size())) {
            elementToFace_[info.elem1].push_back({info.localFace1, idx});
        }
        
        // Add face to element 2 (if exists)
        if (info.elem2 != InvalidIndex && info.elem2 < static_cast<Index>(elementToFace_.size())) {
            elementToFace_[info.elem2].push_back({info.localFace2, idx});
        }
    }
}

void MeshTopology::identifyBoundaryFaces() {
    boundaryFaceIndices_.clear();
    interiorFaceIndices_.clear();
    
    for (const auto& [key, info] : faceToElement_) {
        Index faceIdx = faceToIndex_[key];
        
        if (info.isBoundary) {
            boundaryFaceIndices_.push_back(faceIdx);
        } else {
            interiorFaceIndices_.push_back(faceIdx);
        }
    }
}

Index MeshTopology::numFaces() const {
    return static_cast<Index>(faceToElement_.size());
}

Index MeshTopology::numBoundaryFaces() const {
    return static_cast<Index>(boundaryFaceIndices_.size());
}

Index MeshTopology::numInteriorFaces() const {
    return static_cast<Index>(interiorFaceIndices_.size());
}

bool MeshTopology::isExternalBoundary(Index faceIdx) const {
    for (const auto& [key, info] : faceToElement_) {
        auto it = faceToIndex_.find(key);
        if (it != faceToIndex_.end() && it->second == faceIdx) {
            return info.isBoundary;
        }
    }
    return false;
}

std::pair<Index, Index> MeshTopology::getAdjacentElements(Index faceIdx) const {
    for (const auto& [key, info] : faceToElement_) {
        auto it = faceToIndex_.find(key);
        if (it != faceToIndex_.end() && it->second == faceIdx) {
            return {info.elem1, info.elem2};
        }
    }
    return {InvalidIndex, InvalidIndex};
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
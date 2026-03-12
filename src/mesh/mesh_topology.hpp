#ifndef MPFEM_MESH_TOPOLOGY_HPP
#define MPFEM_MESH_TOPOLOGY_HPP

#include "geometry.hpp"
#include "core/types.hpp"
#include <vector>
#include <unordered_map>
#include <utility>

namespace mpfem {

// Forward declaration
class Mesh;

/**
 * @brief Mesh topology class for neighbor queries.
 * 
 * Provides:
 * - Face to element mapping
 * - Element to face mapping
 * - Internal/external boundary detection
 * - Boundary element to topology face mapping
 * 
 * All queries are O(1) after topology is built.
 */
class MeshTopology {
public:
    /// Face identifier (sorted vertex indices)
    using FaceKey = std::vector<Index>;

    /// Hash function for FaceKey
    struct FaceKeyHash {
        std::size_t operator()(const FaceKey& face) const {
            std::size_t seed = 0;
            for (Index v : face) {
                seed ^= std::hash<Index>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    /// Information about a face's adjacent elements
    struct FaceInfo {
        Index elem1 = InvalidIndex;     ///< First adjacent element
        Index elem2 = InvalidIndex;     ///< Second adjacent element (-1 for external boundary)
        int localFace1 = -1;            ///< Local face index in elem1
        int localFace2 = -1;            ///< Local face index in elem2
        bool isBoundary = true;         ///< True if external boundary face
        std::vector<Index> vertices;    ///< Face vertices (sorted)
    };

    /// Default constructor
    MeshTopology() = default;

    /// Construct from mesh
    explicit MeshTopology(const Mesh* mesh);

    /// Set mesh and build topology
    void setMesh(const Mesh* mesh);

    /// Get the mesh
    const Mesh* mesh() const { return mesh_; }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Get total number of unique faces
    Index numFaces() const { return static_cast<Index>(faceInfoList_.size()); }

    /// Get number of boundary faces (external)
    Index numBoundaryFaces() const { return static_cast<Index>(boundaryFaceIndices_.size()); }

    /// Get number of interior faces
    Index numInteriorFaces() const { return static_cast<Index>(interiorFaceIndices_.size()); }

    // -------------------------------------------------------------------------
    // Face queries (all O(1))
    // -------------------------------------------------------------------------

    /// Check if a face (by index) is an external boundary - O(1)
    bool isExternalBoundary(Index faceIdx) const {
        if (faceIdx >= faceInfoList_.size()) return false;
        return faceInfoList_[faceIdx].isBoundary;
    }

    /// Get adjacent elements for a face - O(1)
    std::pair<Index, Index> getAdjacentElements(Index faceIdx) const {
        if (faceIdx >= faceInfoList_.size()) {
            return {InvalidIndex, InvalidIndex};
        }
        const auto& info = faceInfoList_[faceIdx];
        return {info.elem1, info.elem2};
    }

    /// Get face info by index - O(1)
    const FaceInfo& getFaceInfo(Index faceIdx) const {
        return faceInfoList_[faceIdx];
    }

    /// Get face vertices by index - O(1)
    const std::vector<Index>& getFaceVertices(Index faceIdx) const {
        return faceInfoList_[faceIdx].vertices;
    }

    /// Get boundary elements for a boundary ID
    std::vector<Index> getBoundaryElementsForBoundary(Index boundaryId) const;

    // -------------------------------------------------------------------------
    // Element queries - O(1)
    // -------------------------------------------------------------------------

    /// Get faces for an element
    const std::vector<std::pair<int, Index>>& getElementFaces(Index elemIdx) const {
        return elementToFace_[elemIdx];
    }

    /// Get number of faces for an element
    int numElementFaces(Index elemIdx) const {
        return static_cast<int>(elementToFace_[elemIdx].size());
    }

    // -------------------------------------------------------------------------
    // Boundary face queries
    // -------------------------------------------------------------------------

    /// Get all boundary face indices
    const std::vector<Index>& boundaryFaces() const { return boundaryFaceIndices_; }

    /// Get all interior face indices
    const std::vector<Index>& interiorFaces() const { return interiorFaceIndices_; }

    /// Get boundary face index by boundary element index - O(1)
    /// Returns InvalidIndex if not found
    Index getBoundaryFaceIndex(Index bdrElemIdx) const {
        auto it = bdrElementToFace_.find(bdrElemIdx);
        return (it != bdrElementToFace_.end()) ? it->second : InvalidIndex;
    }

    /// Get boundary element index for a boundary face - O(1)
    /// Returns InvalidIndex if face is not a boundary face
    Index getBoundaryElementIndex(Index faceIdx) const {
        auto it = faceToBdrElement_.find(faceIdx);
        return (it != faceToBdrElement_.end()) ? it->second : InvalidIndex;
    }

private:
    void build();
    void buildFaceToElementMap();
    void buildElementToFaceMap();
    void identifyBoundaryFaces();
    void buildBoundaryElementMapping();

    const Mesh* mesh_ = nullptr;

    /// Face information stored by index - O(1) access
    std::vector<FaceInfo> faceInfoList_;
    
    /// Face key to index mapping (for building phase)
    std::unordered_map<FaceKey, Index, FaceKeyHash> faceKeyToIndex_;
    
    /// Element to face mapping: elementToFace_[elemIdx] = [(localFaceIdx, faceIdx), ...]
    std::vector<std::vector<std::pair<int, Index>>> elementToFace_;
    
    /// Boundary face indices
    std::vector<Index> boundaryFaceIndices_;
    
    /// Interior face indices
    std::vector<Index> interiorFaceIndices_;
    
    /// Boundary element to face mapping
    std::unordered_map<Index, Index> bdrElementToFace_;
    
    /// Face to boundary element mapping (reverse of above)
    std::unordered_map<Index, Index> faceToBdrElement_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_TOPOLOGY_HPP

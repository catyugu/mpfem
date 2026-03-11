#ifndef MPFEM_MESH_TOPOLOGY_HPP
#define MPFEM_MESH_TOPOLOGY_HPP

#include "geometry.hpp"
#include "core/types.hpp"
#include <vector>
#include <unordered_map>
#include <utility>
#include <optional>

namespace mpfem {

// Forward declaration
class Mesh;

/**
 * @brief Mesh topology class for neighbor queries.
 * 
 * Provides:
 * - Element to face mapping
 * - Face to element mapping (for internal/external boundary detection)
 * - Boundary element to adjacent element mapping
 * 
 * A face is identified by its sorted vertex list (canonical representation).
 */
class MeshTopology {
public:
    /// Face identifier (sorted vertex indices)
    using FaceId = std::vector<Index>;

    /// Hash function for FaceId
    struct FaceIdHash {
        std::size_t operator()(const FaceId& face) const {
            std::size_t seed = 0;
            for (Index v : face) {
                seed ^= std::hash<Index>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    /// Information about a face's adjacent elements
    struct FaceInfo {
        Index elem1 = -1;       ///< First adjacent element
        Index elem2 = -1;       ///< Second adjacent element (-1 for external boundary)
        int face1Idx = -1;      ///< Local face index in elem1
        int face2Idx = -1;      ///< Local face index in elem2
        Index bdrElemIdx = -1;  ///< Boundary element index (if external boundary)
    };

    /// Default constructor
    MeshTopology() = default;

    /// Build topology from a mesh
    explicit MeshTopology(const Mesh& mesh);

    /// Build/rebuild topology
    void build(const Mesh& mesh);

    // -------------------------------------------------------------------------
    // Face queries
    // -------------------------------------------------------------------------

    /// Check if a face is an external boundary (only one adjacent element)
    bool isExternalBoundary(const FaceId& face) const;

    /// Check if a face is an internal boundary (two adjacent elements)
    bool isInternalBoundary(const FaceId& face) const;

    /// Get face info
    std::optional<FaceInfo> getFaceInfo(const FaceId& face) const;

    /// Get number of unique faces
    Index numFaces() const { return static_cast<Index>(faceMap_.size()); }

    /// Get all external boundary faces
    std::vector<FaceId> getExternalBoundaryFaces() const;

    /// Get all internal boundary faces
    std::vector<FaceId> getInternalBoundaryFaces() const;

    // -------------------------------------------------------------------------
    // Element-face queries
    // -------------------------------------------------------------------------

    /// Get faces of an element
    const std::vector<FaceId>& getElementFaces(Index elemIdx) const;

    /// Get local face index for an element's face
    int getElementFaceIndex(Index elemIdx, const FaceId& face) const;

    // -------------------------------------------------------------------------
    // Face-element queries
    // -------------------------------------------------------------------------

    /// Get adjacent elements of a face
    std::pair<Index, Index> getFaceElements(const FaceId& face) const;

    /// Get the boundary element for an external face
    Index getBoundaryElement(const FaceId& face) const;

    // -------------------------------------------------------------------------
    // Boundary element queries
    // -------------------------------------------------------------------------

    /// Get the adjacent volume element for a boundary element
    Index getBdrAdjacentElement(Index bdrElemIdx) const;

    /// Get the local face index for a boundary element
    int getBdrLocalFaceIndex(Index bdrElemIdx) const;

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Print topology statistics
    void printStats() const;

    /// Get number of external boundary faces
    Index numExternalBoundaryFaces() const;

    /// Get number of internal boundary faces
    Index numInternalBoundaryFaces() const;

private:
    /// Create canonical face ID (sorted vertices)
    static FaceId makeFaceId(std::vector<Index> vertices);

    /// Map from face to its info
    std::unordered_map<FaceId, FaceInfo, FaceIdHash> faceMap_;

    /// Map from element to its faces
    std::vector<std::vector<FaceId>> elementFaces_;

    /// Map from boundary element index to (element, localFaceIdx)
    std::vector<std::pair<Index, int>> bdrElementInfo_;
};

// =============================================================================
// Inline implementations
// =============================================================================

inline MeshTopology::FaceId MeshTopology::makeFaceId(std::vector<Index> vertices) {
    std::sort(vertices.begin(), vertices.end());
    return vertices;
}

inline bool MeshTopology::isExternalBoundary(const FaceId& face) const {
    auto it = faceMap_.find(face);
    if (it == faceMap_.end()) return false;
    return it->second.elem2 < 0;
}

inline bool MeshTopology::isInternalBoundary(const FaceId& face) const {
    auto it = faceMap_.find(face);
    if (it == faceMap_.end()) return false;
    return it->second.elem2 >= 0;
}

inline std::optional<MeshTopology::FaceInfo> MeshTopology::getFaceInfo(const FaceId& face) const {
    auto it = faceMap_.find(face);
    if (it == faceMap_.end()) return std::nullopt;
    return it->second;
}

inline const std::vector<MeshTopology::FaceId>& MeshTopology::getElementFaces(Index elemIdx) const {
    return elementFaces_[elemIdx];
}

inline std::pair<Index, Index> MeshTopology::getFaceElements(const FaceId& face) const {
    auto it = faceMap_.find(face);
    if (it == faceMap_.end()) return {-1, -1};
    return {it->second.elem1, it->second.elem2};
}

inline Index MeshTopology::getBoundaryElement(const FaceId& face) const {
    auto it = faceMap_.find(face);
    if (it == faceMap_.end()) return -1;
    return it->second.bdrElemIdx;
}

inline Index MeshTopology::getBdrAdjacentElement(Index bdrElemIdx) const {
    if (bdrElemIdx < 0 || bdrElemIdx >= static_cast<Index>(bdrElementInfo_.size())) {
        return -1;
    }
    return bdrElementInfo_[bdrElemIdx].first;
}

inline int MeshTopology::getBdrLocalFaceIndex(Index bdrElemIdx) const {
    if (bdrElemIdx < 0 || bdrElemIdx >= static_cast<Index>(bdrElementInfo_.size())) {
        return -1;
    }
    return bdrElementInfo_[bdrElemIdx].second;
}

}  // namespace mpfem

#endif  // MPFEM_MESH_TOPOLOGY_HPP

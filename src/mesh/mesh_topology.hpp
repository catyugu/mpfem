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
    Index numFaces() const;

    /// Get number of boundary faces (external)
    Index numBoundaryFaces() const;

    /// Get number of interior faces
    Index numInteriorFaces() const;

    // -------------------------------------------------------------------------
    // Face queries
    // -------------------------------------------------------------------------

    /// Check if a face (by index) is an external boundary
    bool isExternalBoundary(Index faceIdx) const;

    /// Get adjacent elements for a face
    std::pair<Index, Index> getAdjacentElements(Index faceIdx) const;

    /// Get boundary elements for a boundary ID
    std::vector<Index> getBoundaryElementsForBoundary(Index boundaryId) const;

    // -------------------------------------------------------------------------
    // Element queries
    // -------------------------------------------------------------------------

    /// Get faces for an element
    const std::vector<std::pair<int, Index>>& getElementFaces(Index elemIdx) const {
        return elementToFace_[elemIdx];
    }

private:
    void build();
    void buildFaceToElementMap();
    void buildElementToFaceMap();
    void identifyBoundaryFaces();

    const Mesh* mesh_ = nullptr;

    /// Face to element mapping
    std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> faceToElement_;
    
    /// Face to vertex list
    std::unordered_map<FaceKey, std::vector<Index>, FaceKeyHash> faceVertices_;
    
    /// Face key to index mapping
    std::unordered_map<FaceKey, Index, FaceKeyHash> faceToIndex_;
    
    /// Element to face mapping: elementToFace_[elemIdx] = [(localFaceIdx, faceIdx), ...]
    std::vector<std::vector<std::pair<int, Index>>> elementToFace_;
    
    /// Boundary face indices
    std::vector<Index> boundaryFaceIndices_;
    
    /// Interior face indices
    std::vector<Index> interiorFaceIndices_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_TOPOLOGY_HPP
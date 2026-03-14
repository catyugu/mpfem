#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "geometry.hpp"
#include "vertex.hpp"
#include "element.hpp"
#include "core/types.hpp"
#include "core/exception.hpp"
#include <vector>
#include <memory>
#include <set>
#include <span>
#include <unordered_map>

namespace mpfem {

/**
 * @brief Mesh class storing vertices, elements, and topology.
 * 
 * This class manages:
 * - Vertex coordinates
 * - Volume elements (tetrahedra, hexahedra)
 * - Boundary elements (triangles, quads)
 * - Domain and boundary attributes
 * - Mesh topology for internal/external boundary detection
 */
class Mesh {
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
    Mesh() = default;

    /// Construct with pre-allocated sizes
    Mesh(int dim, Index numVertices, Index numElements, Index numBdrElements = 0);

    // -------------------------------------------------------------------------
    // Dimension and size
    // -------------------------------------------------------------------------

    /// Get spatial dimension
    int dim() const { return dim_; }

    /// Set spatial dimension
    void setDim(int dim);

    /// Get number of vertices
    Index numVertices() const { return static_cast<Index>(vertices_.size()); }

    /// Get number of volume elements
    Index numElements() const { return static_cast<Index>(elements_.size()); }

    /// Get number of boundary elements
    Index numBdrElements() const { return static_cast<Index>(bdrElements_.size()); }

    // -------------------------------------------------------------------------
    // Vertex access
    // -------------------------------------------------------------------------

    /// Get vertex by index
    const Vertex& vertex(Index i) const { return vertices_[i]; }
    Vertex& vertex(Index i) { return vertices_[i]; }

    /// Get all vertices
    const std::vector<Vertex>& vertices() const { return vertices_; }
    std::vector<Vertex>& vertices() { return vertices_; }

    /// Add a vertex
    void addVertex(const Vertex& v);
    void addVertex(Vertex&& v);
    
    /// Add a vertex from coordinates
    Index addVertex(Real x, Real y = 0.0, Real z = 0.0);

    /// Reserve space for vertices
    void reserveVertices(Index n);

    // -------------------------------------------------------------------------
    // Volume element access
    // -------------------------------------------------------------------------

    /// Get element by index
    const Element& element(Index i) const { return elements_[i]; }
    Element& element(Index i) { return elements_[i]; }

    /// Get all elements
    const std::vector<Element>& elements() const { return elements_; }
    std::vector<Element>& elements() { return elements_; }

    /// Add an element
    void addElement(const Element& e);
    void addElement(Element&& e);
    Index addElement(Geometry geom, std::span<const Index> vertices, Index attr = 0, int order = 1);
    Index addElement(Geometry geom, const std::vector<Index>& vertices, Index attr = 0, int order = 1);

    /// Reserve space for elements
    void reserveElements(Index n);

    // -------------------------------------------------------------------------
    // Boundary element access
    // -------------------------------------------------------------------------

    /// Get boundary element by index
    const Element& bdrElement(Index i) const { return bdrElements_[i]; }
    Element& bdrElement(Index i) { return bdrElements_[i]; }

    /// Get all boundary elements
    const std::vector<Element>& bdrElements() const { return bdrElements_; }
    std::vector<Element>& bdrElements() { return bdrElements_; }

    /// Add a boundary element
    void addBdrElement(const Element& e);
    void addBdrElement(Element&& e);
    Index addBdrElement(Geometry geom, std::span<const Index> vertices, Index attr = 0, int order = 1);
    Index addBdrElement(Geometry geom, const std::vector<Index>& vertices, Index attr = 0, int order = 1);

    /// Reserve space for boundary elements
    void reserveBdrElements(Index n);

    // -------------------------------------------------------------------------
    // Domain and boundary attribute queries
    // -------------------------------------------------------------------------

    /// Get all unique domain IDs
    std::set<Index> domainIds() const;

    /// Get all unique boundary IDs
    std::set<Index> boundaryIds() const;

    /// Get elements in a specific domain
    std::vector<Index> elementsForDomain(Index domainId) const;

    /// Get boundary elements with a specific boundary ID
    std::vector<Index> bdrElementsForBoundary(Index boundaryId) const;

    // -------------------------------------------------------------------------
    // Topology queries (for internal/external boundary detection)
    // -------------------------------------------------------------------------

    /// Build mesh topology (call after mesh is fully loaded)
    void buildTopology();

    /// Check if topology has been built
    bool hasTopology() const { return topologyBuilt_; }

    /// Check if a boundary element is an external boundary (not internal interface)
    /// Returns true if on external boundary, false if internal interface
    bool isExternalBoundary(Index bdrElemIdx) const {
        if (!topologyBuilt_) return true;  // Without topology, assume all are external
        auto it = bdrElementToFace_.find(bdrElemIdx);
        if (it == bdrElementToFace_.end()) return true;
        return faceInfoList_[it->second].isBoundary;
    }

    /// Check if a boundary ID (attribute) is an external boundary
    /// This is efficient: same boundary ID means same external/internal status
    bool isExternalBoundaryId(Index bdrId) const {
        if (!topologyBuilt_) return true;
        auto it = bdrIdExternalCache_.find(bdrId);
        return (it != bdrIdExternalCache_.end()) ? it->second : true;
    }

    /// Get face info by index
    const FaceInfo& getFaceInfo(Index faceIdx) const { return faceInfoList_[faceIdx]; }

    /// Get total number of unique faces
    Index numFaces() const { return static_cast<Index>(faceInfoList_.size()); }

    /// Get number of boundary faces (external)
    Index numBoundaryFaces() const { return static_cast<Index>(boundaryFaceIndices_.size()); }

    /// Get number of interior faces
    Index numInteriorFaces() const { return static_cast<Index>(interiorFaceIndices_.size()); }

    /// Get boundary face index by boundary element index
    Index getBoundaryFaceIndex(Index bdrElemIdx) const {
        auto it = bdrElementToFace_.find(bdrElemIdx);
        return (it != bdrElementToFace_.end()) ? it->second : InvalidIndex;
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    /// Clear all data
    void clear();

    /// Get bounding box (min, max)
    std::pair<Vector3, Vector3> getBoundingBox() const;

private:
    void buildFaceToElementMap();
    void buildElementToFaceMap();
    void identifyBoundaryFaces();
    void buildBoundaryElementMapping();

    int dim_ = 3;
    std::vector<Vertex> vertices_;
    std::vector<Element> elements_;
    std::vector<Element> bdrElements_;

    // Topology data
    bool topologyBuilt_ = false;
    std::vector<FaceInfo> faceInfoList_;
    std::unordered_map<FaceKey, Index, FaceKeyHash> faceKeyToIndex_;
    std::vector<std::vector<std::pair<int, Index>>> elementToFace_;
    std::vector<Index> boundaryFaceIndices_;
    std::vector<Index> interiorFaceIndices_;
    std::unordered_map<Index, Index> bdrElementToFace_;
    std::unordered_map<Index, bool> bdrIdExternalCache_;  ///< Cache: boundary ID -> isExternal
};

}  // namespace mpfem

#endif  // MPFEM_MESH_HPP
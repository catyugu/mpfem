#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "element.hpp"
#include "geometry.hpp"
#include "vertex.hpp"
#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>


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
        struct FaceKey {
            int nodes[4]; // 假设最多四边形面
            int count;

            // 必须自己实现 == 和固定的 hash 函数 (如 MurmurHash 或简单按位异或)
            bool operator==(const FaceKey& other) const
            {
                if (count != other.count)
                    return false;
                for (int i = 0; i < count; ++i)
                    if (nodes[i] != other.nodes[i])
                        return false;
                return true;
            }
        };

        // 哈希函数规避堆分配
        struct FaceKeyHash {
            std::size_t operator()(const FaceKey& k) const
            {
                std::size_t h = 0;
                for (int i = 0; i < k.count; ++i) {
                    h ^= std::hash<int> {}(k.nodes[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                return h;
            }
        };

        /// Information about a face's adjacent elements
        struct FaceInfo {
            Index elem1 = InvalidIndex; ///< First adjacent element
            Index elem2 = InvalidIndex; ///< Second adjacent element (-1 for external boundary)
            int localFace1 = -1; ///< Local face index in elem1
            int localFace2 = -1; ///< Local face index in elem2
            bool isBoundary = true; ///< True if external boundary face
            std::vector<Index> vertices; ///< Face vertices (sorted)
        };

        struct EdgeInfo {
            Index v0 = InvalidIndex;
            Index v1 = InvalidIndex;
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

        /// Get total number of unique topology edges
        Index numEdges() const { return static_cast<Index>(edgeInfoList_.size()); }

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
        bool isExternalBoundary(Index bdrElemIdx) const
        {
            if (!topologyBuilt_)
                return true; // Without topology, assume all are external
            auto it = bdrElementToFace_.find(bdrElemIdx);
            if (it == bdrElementToFace_.end())
                return true;
            return faceInfoList_[it->second].isBoundary;
        }

        /// Check if a boundary ID (attribute) is an external boundary
        /// This is efficient: same boundary ID means same external/internal status
        bool isExternalBoundaryId(Index bdrId) const
        {
            if (!topologyBuilt_)
                return true;
            auto it = bdrIdExternalCache_.find(bdrId);
            return (it != bdrIdExternalCache_.end()) ? it->second : true;
        }

        /// Get face info by index
        const FaceInfo& getFaceInfo(Index faceIdx) const { return faceInfoList_[faceIdx]; }

        /// Get total number of unique faces
        Index numFaces() const { return static_cast<Index>(faceInfoList_.size()); }

        /// Get global topology vertices used by an element (corner vertices only)
        std::vector<Index> getElementVertices(Index elemIdx) const;

        /// Get global topology edge indices used by an element (local edge order)
        std::span<const Index> getElementEdges(Index elemIdx) const;

        /// Get global topology face indices used by an element (local face order)
        std::span<const Index> getElementFaces(Index elemIdx) const;

        /// Get global topology edge index by two endpoint vertices
        Index edgeIndex(Index a, Index b) const;

        /// Get number of boundary faces (external)
        Index numBoundaryFaces() const { return static_cast<Index>(boundaryFaceIndices_.size()); }

        /// Get number of interior faces
        Index numInteriorFaces() const { return static_cast<Index>(interiorFaceIndices_.size()); }

        /// Get boundary face index by boundary element index
        Index getBoundaryFaceIndex(Index bdrElemIdx) const
        {
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

        // -------------------------------------------------------------------------
        // Corner vertices (topological vertices for high-order meshes)
        // -------------------------------------------------------------------------

        /// Get number of corner vertices (topological vertices)
        /// For linear meshes, this equals numVertices()
        /// For quadratic meshes, this returns only the geometric corners
        Index numCornerVertices() const;

        /// Get corner vertex indices (lazy evaluation, cached)
        /// Returns a sorted list of unique corner vertex indices
        const std::vector<Index>& cornerVertexIndices() const;

        /// Get mapping from vertex index to corner index
        /// Returns InvalidIndex if vertex is not a corner
        Index vertexToCornerIndex(Index vertexIdx) const;

    private:
        static std::uint64_t edgeKey(Index a, Index b);
        void buildEdgeToElementMap();
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
        std::vector<EdgeInfo> edgeInfoList_;
        std::unordered_map<std::uint64_t, Index> edgeKeyToIndex_;
        
        // CSR storage for element-to-edge
        std::vector<Index> elemEdgeOffsets_;
        std::vector<Index> elemEdgeData_;
        
        std::vector<FaceInfo> faceInfoList_;
        std::unordered_map<FaceKey, Index, FaceKeyHash> faceKeyToIndex_;
        
        // CSR storage for element-to-face
        std::vector<Index> elemFaceOffsets_;
        std::vector<Index> elemFaceData_;
        std::vector<Index> boundaryFaceIndices_;
        std::vector<Index> interiorFaceIndices_;
        std::unordered_map<Index, Index> bdrElementToFace_;
        std::unordered_map<Index, bool> bdrIdExternalCache_; ///< Cache: boundary ID -> isExternal

        // Corner vertex data (for high-order meshes)
        std::vector<Index> cornerVertexIndices_; ///< Sorted list of corner vertex indices
        std::vector<Index> cornerVertexMap_; ///< Mapping: vertex index -> corner index (InvalidIndex if not corner)
    };

} // namespace mpfem

#endif // MPFEM_MESH_HPP
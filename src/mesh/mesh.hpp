#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "core/exception.hpp"
#include "core/geometry.hpp"
#include "core/types.hpp"
#include "element.hpp"
#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

namespace mpfem {

    /**
     * @brief Stack-allocated face key for topology building.
     *
     * Used during buildFaceToElementMap() to avoid millions of heap allocations
     * when using std::vector<Index>. Face node count is bounded (max 4 for quad),
     * so this fixed-size array is always sufficient.
     */
    struct FaceKey {
        static constexpr int MAX_FACE_NODES = 4;
        Index nodes[MAX_FACE_NODES]; // Sorted node indices
        int count;

        FaceKey() : count(0) { }

        /// Initialize from sorted node span
        void set(std::span<const Index> sorted_nodes)
        {
            count = sorted_nodes.size();
            for (int i = 0; i < count; ++i)
                nodes[i] = sorted_nodes[i];
        }

        /// Lexicographic comparison for sorting
        bool operator<(const FaceKey& o) const
        {
            if (count != o.count)
                return count < o.count;
            for (int i = 0; i < count; ++i) {
                if (nodes[i] != o.nodes[i])
                    return nodes[i] < o.nodes[i];
            }
            return false;
        }
        bool operator==(const FaceKey& o) const
        {
            if (count != o.count)
                return false;
            for (int i = 0; i < count; ++i)
                if (nodes[i] != o.nodes[i])
                    return false;
            return true;
        }
    };

    /**
     * @brief Mesh class storing nodes, elements, and topology.
     *
     * This class manages:
     * - Vertex coordinates (interleaved [x,y,z,x,y,z...] for C API zero-copy)
     * - Volume elements (tetrahedra, hexahedra)
     * - Boundary elements (triangles, quads)
     * - Domain and boundary attributes
     * - Mesh topology for internal/external boundary detection
     *
     * Data Layout:
     * - Nodes: interleaved coords_[dim * nodeIdx + d] (C API zero-copy)
     * - Elements: CSR with elementOffsets_ + elementNodes_
     * - Faces: CSR with faceOffsets_ + faceNodes_
     * - Edges: sorted edgeInfoList_ for binary search
     */
    class Mesh {
    public:
        /// Edge information
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

        /// Get number of nodes
        Index numNodes() const { return static_cast<Index>(coords_.size()) / dim_; }

        /// Get total number of unique topology edges
        Index numEdges() const { return static_cast<Index>(edgeInfoList_.size()); }

        // -------------------------------------------------------------------------
        // Vertex access (interleaved [x,y,z,...] for C API zero-copy)
        // -------------------------------------------------------------------------

        /// Add a node from coordinates
        Index addNode(Real x, Real y = 0.0, Real z = 0.0);

        /// Reserve space for nodes
        void reserveNodes(Index n);

        /// Node coordinate accessors - interleaved storage for zero-copy C API
        Real nodeX(Index i) const { return coords_[i * dim_]; }
        Real nodeY(Index i) const { return coords_[i * dim_ + 1]; }
        Real nodeZ(Index i) const { return coords_[i * dim_ + 2]; }

        /// Get raw coords pointer for zero-copy C API (e.g., VTK, CGNS)
        const Real* nodeCoordsData() const { return coords_.data(); }

        // -------------------------------------------------------------------------
        // Volume element access
        // -------------------------------------------------------------------------

        /// Get element by index (returns by value as a view)
        Element element(Index i) const;

        /// Get number of volume elements
        Index numElements() const { return static_cast<Index>(elementGeoms_.size()); }

        /// Add an element
        Index addElement(Geometry geom, std::span<const Index> nodes, Index attr = 0, int order = 1);
        Index addElement(Geometry geom, const std::vector<Index>& nodes, Index attr = 0, int order = 1);

        /// Reserve space for elements
        void reserveElements(Index n);

        // -------------------------------------------------------------------------
        // Boundary element access
        // -------------------------------------------------------------------------

        /// Get boundary element by index (returns by value as a view)
        Element bdrElement(Index i) const;

        /// Get number of boundary elements
        Index numBdrElements() const { return static_cast<Index>(bdrElementGeoms_.size()); }

        /// Add a boundary element
        Index addBdrElement(Geometry geom, std::span<const Index> nodes, Index attr = 0, int order = 1);
        Index addBdrElement(Geometry geom, const std::vector<Index>& nodes, Index attr = 0, int order = 1);

        /// Reserve space for boundary elements
        void reserveBdrElements(Index n);

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
            return faceIsBoundary(it->second);
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

        /// Get total number of unique faces
        Index numFaces() const { return static_cast<Index>(faceOffsets_.size()) - 1; }

        /// Number of nodes for a face
        Index numFaceNodes(Index faceIdx) const { return faceOffsets_[faceIdx + 1] - faceOffsets_[faceIdx]; }

        /// Face nodes (CSR F2N)
        std::span<const Index> faceNodes(Index faceIdx) const
        {
            return {&faceNodes_[faceOffsets_[faceIdx]], static_cast<size_t>(numFaceNodes(faceIdx))};
        }
        Index faceNeighborElem1(Index faceIdx) const { return faceElem1_[faceIdx]; }
        Index faceNeighborElem2(Index faceIdx) const { return faceElem2_[faceIdx]; }
        int faceLocalIndex1(Index faceIdx) const { return faceLocal1_[faceIdx]; }
        int faceLocalIndex2(Index faceIdx) const { return faceLocal2_[faceIdx]; }
        bool faceIsBoundary(Index faceIdx) const { return faceBoundary_[faceIdx] != 0; }

        /// Get global topology edge indices used by an element (local edge order)
        std::span<const Index> elementEdges(Index elemIdx) const;

        /// Get global topology face indices used by an element (local face order)
        std::span<const Index> elementFaces(Index elemIdx) const;

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

        // -------------------------------------------------------------------------
        // Corner vertices (topological vertices for high-order meshes)
        // -------------------------------------------------------------------------

    private:
        void buildEdgeToElementMap();
        void buildFaceToElementMap();
        void buildElementToFaceMap();
        void identifyBoundaryFaces();
        void buildBoundaryElementMapping();

        // Mesh dimension
        int dim_ = 3;

        // Interleaved node coordinates [x0,y0,z0,x1,y1,z1,...] for zero-copy C API
        std::vector<Real> coords_;

        // Flattened volume element storage
        std::vector<Geometry> elementGeoms_;
        std::vector<Index> elementAttributes_;
        std::vector<int> elementOrders_;
        std::vector<Index> elementOffsets_;
        std::vector<Index> elementNodes_;

        // Flattened boundary element storage
        std::vector<Geometry> bdrElementGeoms_;
        std::vector<Index> bdrElementAttributes_;
        std::vector<int> bdrElementOrders_;
        std::vector<Index> bdrElementOffsets_;
        std::vector<Index> bdrElementNodes_;

        // Topology data
        bool topologyBuilt_ = false;
        std::vector<EdgeInfo> edgeInfoList_; // Sorted by (v0, v1) for binary search

        // CSR storage for element-to-edge
        std::vector<Index> elemEdgeOffsets_;
        std::vector<Index> elemEdgeData_;

        // CSR storage for element-to-face
        std::vector<Index> elemFaceOffsets_;
        std::vector<Index> elemFaceData_;
        std::vector<Index> boundaryFaceIndices_;
        std::vector<Index> interiorFaceIndices_;
        std::unordered_map<Index, Index> bdrElementToFace_;
        std::unordered_map<Index, bool> bdrIdExternalCache_; ///< Cache: boundary ID -> isExternal

        // CSR storage for faces (F2N) - METIS compatible
        std::vector<Index> faceOffsets_; // CSR row pointers (numFaces+1)
        std::vector<Index> faceNodes_; // Flattened face node indices
        std::vector<Index> faceElem1_; // First adjacent element
        std::vector<Index> faceElem2_; // Second adjacent element (InvalidIndex for boundary)
        std::vector<int> faceLocal1_; // Local face index in elem1
        std::vector<int> faceLocal2_; // Local face index in elem2
        std::vector<char> faceBoundary_; // char (0=interior, 1=boundary)

        // Sorted face keys for binary search in boundary element matching
        // Uses stack-allocated FaceKey to avoid heap allocation during topology building
        std::vector<std::pair<FaceKey, Index>> sortedFaceKeys_; // Sorted by FaceKey for binary search
    };

} // namespace mpfem

#endif // MPFEM_MESH_HPP
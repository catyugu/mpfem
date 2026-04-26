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
     * @brief Mesh class storing nodes, elements, and topology.
     *
     * This class manages:
     * - Vertex coordinates (Structure of Arrays for METIS compatibility)
     * - Volume elements (tetrahedra, hexahedra)
     * - Boundary elements (triangles, quads)
     * - Domain and boundary attributes
     * - Mesh topology for internal/external boundary detection
     *
     * Data Layout:
     * - Nodes: SoA with separate x_, y_, z_ arrays (METIS-compatible)
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
        Index numNodes() const { return static_cast<Index>(x_.size()); }

        /// Get total number of unique topology edges
        Index numEdges() const { return static_cast<Index>(edgeInfoList_.size()); }

        // -------------------------------------------------------------------------
        // Vertex access (SoA for METIS compatibility)
        // -------------------------------------------------------------------------

        /// Add a node from coordinates
        Index addNode(Real x, Real y = 0.0, Real z = 0.0);

        /// Reserve space for nodes
        void reserveNodes(Index n);

        /// Node coordinate accessors (METIS-compatible separate arrays)
        Real nodeX(Index i) const { return x_[i]; }
        Real nodeY(Index i) const { return y_[i]; }
        Real nodeZ(Index i) const { return z_[i]; }

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

        /// Face queries (CSR F2N)
        Index numFaceNodes(Index faceIdx) const { return faceOffsets_[faceIdx + 1] - faceOffsets_[faceIdx]; }
        std::span<const Index> getFaceNodes(Index faceIdx) const
        {
            return {&faceNodes_[faceOffsets_[faceIdx]], static_cast<size_t>(numFaceNodes(faceIdx))};
        }
        Index faceNeighborElem1(Index faceIdx) const { return faceElem1_[faceIdx]; }
        Index faceNeighborElem2(Index faceIdx) const { return faceElem2_[faceIdx]; }
        int faceLocalIndex1(Index faceIdx) const { return faceLocal1_[faceIdx]; }
        int faceLocalIndex2(Index faceIdx) const { return faceLocal2_[faceIdx]; }
        bool faceIsBoundary(Index faceIdx) const { return faceBoundary_[faceIdx] != 0; }

        /// For topology building - set face data during buildTopology
        void setFaceData(Index faceIdx, Index elem1, Index elem2, int local1, int local2, bool isBdr,
            const std::vector<Index>& nodes);
        void appendFace(const std::vector<Index>& nodes, Index elem1, Index elem2, int local1, int local2, bool isBdr);

        /// Get global topology vertices used by an element
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

    private:
        void buildEdgeToElementMap();
        void buildFaceToElementMap();
        void buildElementToFaceMap();
        void identifyBoundaryFaces();
        void buildBoundaryElementMapping();

        // Mesh dimension
        int dim_ = 3;

        // Structure of Arrays for nodes (METIS-compatible)
        std::vector<Real> x_; // Node x coordinates
        std::vector<Real> y_; // Node y coordinates
        std::vector<Real> z_; // Node z coordinates

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
        // FaceKey: pair of (sorted node indices, faceIdx)
        using FaceKeyType = std::vector<Index>;
        std::vector<std::pair<FaceKeyType, Index>> sortedFaceKeys_; // Sorted by node vector for binary search
    };

} // namespace mpfem

#endif // MPFEM_MESH_HPP
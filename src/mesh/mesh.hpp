#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"
#include <span>
#include <unordered_map>
#include <vector>

namespace mpfem {

    /**
     * @brief EntityView - a non-owning view of a mesh entity (vertex/edge/face/cell).
     *
     * This is the unified entity representation in the CW Complex approach.
     * All mesh entities (0D vertices, 1D edges, 2D faces, 3D cells) are accessed
     * through this same view type - the dimension and geometry type distinguish them.
     */
    struct EntityView {
        Geometry geometry = Geometry::Invalid;
        std::span<const Index> vertices;
        std::span<const Index> nodes;
        Index attribute = 0;
        int order = 1;

        int dim() const { return geom::dim(geometry); }
        int numVertices() const { return static_cast<int>(vertices.size()); }
        int numNodes() const { return static_cast<int>(nodes.size()); }
        int numEdges() const { return geom::numEdges(geometry); }
        int numFaces() const { return geom::numFaces(geometry); }
        int numFacets() const { return geom::numFacets(geometry); }

        bool isVolume() const { return geom::isVolume(geometry); }
        bool isSurface() const { return geom::isSurface(geometry); }

        Index vertex(int i) const { return vertices[i]; }

        std::pair<Index, Index> edgeVertices(int edgeIdx) const
        {
            auto local = geom::edgeVertices(geometry, edgeIdx);
            return {vertices[local.first], vertices[local.second]};
        }

        std::vector<Index> faceVertices(int faceIdx) const
        {
            std::vector<Index> result;
            auto localVerts = geom::faceVertices(geometry, faceIdx);
            result.reserve(localVerts.size());
            for (int lv : localVerts) {
                result.push_back(vertices[lv]);
            }
            return result;
        }

        std::vector<Index> facetVertices(int facetIdx) const
        {
            std::vector<Index> result;
            auto localVerts = geom::facetVertices(geometry, facetIdx);
            result.reserve(localVerts.size());
            for (int lv : localVerts) {
                result.push_back(vertices[lv]);
            }
            return result;
        }

        Geometry faceGeometry(int faceIdx) const { return geom::faceGeometry(geometry, faceIdx); }
        Geometry facetGeometry(int facetIdx) const { return geom::facetGeometry(geometry, facetIdx); }
    };

    /**
     * @brief Represents all mesh entities of a given topological dimension.
     *
     * This is the core building block of the CW Complex approach - all mesh entities
     * (vertices, edges, faces, cells) are stored in their respective strata by dimension.
     */
    struct EntityStratum {
        int dim = -1;                          ///< Topological dimension (0=vertex, 1=edge, 2=face, 3=cell)
        std::vector<Geometry> geometries;      ///< Geometry type per entity
        std::vector<Index> offsets;            ///< CSR row pointers (size = count + 1)
        std::vector<Index> nodes;              ///< Flattened node indices (CSR data)
        std::vector<Index> attributes;         ///< Physical/domain IDs
        std::vector<int> orders;               ///< Polynomial order for curved elements

        Index count() const { return static_cast<Index>(geometries.size()); }
    };

    /**
     * @brief Core mesh topology class
     *
     * Manages:
     * - Vertex coordinates (interleaved [x,y,z,x,y,z...] for C API zero-copy)
     * - All mesh entities organized by topological dimension (stratum)
     * - Domain and boundary attributes
     * - Mesh topology for internal/external boundary detection
     * - Edge and face orientations for H(curl) and H(div) spaces
     *
     * Data Layout:
     * - Nodes: interleaved coords_[dim * nodeIdx + d] (C API zero-copy)
     * - Stratum[dim]: CSR with offsets_ + nodes, one entry per entity
     * - Edges: flat edgeVertices_ [v0, v1, v0, v1, ...] with v0 < v1
     * - Faces: CSR with faceOffsets_ + faceNodes_
     */
    class Mesh {
    public:
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
        // Unified entity access by topological dimension
        // -------------------------------------------------------------------------

        /// Get number of entities in a given topological dimension
        /// dim=0: vertices, dim=1: edges, dim=2: faces, dim=3: cells (volume elements)
        Index numEntities(int dim) const { return strata_[dim].count(); }

        /// Get entity by dimension and local index within that dimension
        EntityView entity(int dim, Index id) const;

        /// Add an entity to a given dimension stratum
        Index addEntity(int dim, Geometry geom, std::span<const Index> nodes, Index attr = 0, int order = 1);
        Index addEntity(int dim, Geometry geom, const std::vector<Index>& nodes, Index attr = 0, int order = 1);

        /// Reserve space for entities in a given dimension
        void reserveEntities(int dim, Index n);

        // -------------------------------------------------------------------------
        // Volume element access (delegates to strata_[meshDim])
        // -------------------------------------------------------------------------

        /// Get element by index (returns by value as a view)
        EntityView element(Index i) const { return entity(dim_, i); }

        /// Get number of volume elements
        Index numElements() const { return numEntities(dim_); }

        /// Add an element (delegates to addEntity for current mesh dimension)
        Index addElement(Geometry geom, std::span<const Index> nodes, Index attr = 0, int order = 1)
        {
            return addEntity(dim_, geom, nodes, attr, order);
        }
        Index addElement(Geometry geom, const std::vector<Index>& nodes, Index attr = 0, int order = 1)
        {
            return addEntity(dim_, geom, nodes, attr, order);
        }

        /// Reserve space for elements
        void reserveElements(Index n) { reserveEntities(dim_, n); }

        // -------------------------------------------------------------------------
        // Boundary element access (delegates to strata_[meshDim-1])
        // -------------------------------------------------------------------------

        /// Get boundary element by index (returns by value as a view)
        EntityView bdrElement(Index i) const { return entity(dim_ - 1, i); }

        /// Get number of boundary elements
        Index numBdrElements() const { return numEntities(dim_ - 1); }

        /// Add a boundary element (delegates to addEntity for meshDim-1)
        Index addBdrElement(Geometry geom, std::span<const Index> nodes, Index attr = 0, int order = 1)
        {
            return addEntity(dim_ - 1, geom, nodes, attr, order);
        }
        Index addBdrElement(Geometry geom, const std::vector<Index>& nodes, Index attr = 0, int order = 1)
        {
            return addEntity(dim_ - 1, geom, nodes, attr, order);
        }

        /// Reserve space for boundary elements
        void reserveBdrElements(Index n) { reserveEntities(dim_ - 1, n); }

        // -------------------------------------------------------------------------
        // Topology queries
        // -------------------------------------------------------------------------

        /// Build mesh topology (call after mesh is fully loaded)
        void buildTopology();

        /// Check if topology has been built
        bool hasTopology() const { return topologyBuilt_; }

        /// Clear all data
        void clear();

        // -------------------------------------------------------------------------
        // Edge topology and orientations
        // -------------------------------------------------------------------------

        /// Get total number of unique topology edges
        Index numEdges() const { return static_cast<Index>(edgeVertices_.size()) / 2; }

        /// Get edge vertices - returns span of 2 indices [v0, v1] with v0 < v1 (global positive direction)
        std::span<const Index> edgeVertices(Index edgeIdx) const
        {
            return {&edgeVertices_[edgeIdx * 2], 2};
        }

        /// Get global topology edge index by two endpoint vertices
        Index edgeIndex(Index a, Index b) const;

        /// Get global topology edge indices used by an element (local edge order)
        std::span<const Index> elementEdges(Index elemIdx) const;

        /// Get edge orientations for an element: 1 = same direction as global edge, -1 = reversed
        std::span<const int> elementEdgeOrientations(Index elemIdx) const;

        // -------------------------------------------------------------------------
        // Face topology and orientations
        // -------------------------------------------------------------------------

        /// Get total number of unique faces
        Index numFaces() const { return static_cast<Index>(faceOffsets_.size()) - 1; }

        /// Number of nodes for a face
        Index numFaceNodes(Index faceIdx) const { return faceOffsets_[faceIdx + 1] - faceOffsets_[faceIdx]; }

        /// Face nodes (CSR F2N)
        std::span<const Index> faceNodes(Index faceIdx) const
        {
            return {&faceNodes_[faceOffsets_[faceIdx]], static_cast<size_t>(numFaceNodes(faceIdx))};
        }

        /// Get global topology face indices used by an element (local face order)
        std::span<const Index> elementFaces(Index elemIdx) const;

        /// Get face orientations for an element: 1 = face normal matches global, -1 = reversed
        std::span<const int> elementFaceOrientations(Index elemIdx) const;

        /// First adjacent element to a face (InvalidIndex for boundary faces)
        Index faceNeighborElem1(Index faceIdx) const { return faceElem1_[faceIdx]; }

        /// Second adjacent element to a face (InvalidIndex for boundary faces)
        Index faceNeighborElem2(Index faceIdx) const { return faceElem2_[faceIdx]; }

        /// Local face index in elem1
        int faceLocalIndex1(Index faceIdx) const { return faceLocal1_[faceIdx]; }

        /// Local face index in elem2
        int faceLocalIndex2(Index faceIdx) const { return faceLocal2_[faceIdx]; }

        /// Check if face is on external boundary
        bool faceIsBoundary(Index faceIdx) const { return faceBoundary_[faceIdx] != 0; }

        // -------------------------------------------------------------------------
        // Boundary topology queries
        // -------------------------------------------------------------------------

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

    private:
        void buildEdgeTopology();
        void buildFaceTopology();

        // Mesh dimension
        int dim_ = 3;

        // Interleaved node coordinates [x0,y0,z0,x1,y1,z1,...] for zero-copy C API
        std::vector<Real> coords_;

        // Stratum-based storage: entities organized by topological dimension
        // dim 0: vertices (0D), dim 1: edges (1D), dim 2: faces (2D), dim 3: cells (3D)
        std::array<EntityStratum, 4> strata_;

        // Topology data
        bool topologyBuilt_ = false;

        // Edge topology: flat array [v0, v1, v0, v1, ...] with v0 < v1 (global positive direction)
        std::vector<Index> edgeVertices_;

        // CSR storage for element-to-edge
        std::vector<Index> elemEdgeOffsets_;
        std::vector<Index> elemEdgeData_;
        std::vector<int> elemEdgeOrientations_; // 1 = same direction, -1 = reversed

        // CSR storage for element-to-face
        std::vector<Index> elemFaceOffsets_;
        std::vector<Index> elemFaceData_;
        std::vector<int> elemFaceOrientations_; // 1 = normal matches, -1 = reversed

        // CSR storage for faces (F2N) - METIS compatible
        std::vector<Index> faceOffsets_; // CSR row pointers (numFaces+1)
        std::vector<Index> faceNodes_; // Flattened face node indices
        std::vector<Index> faceElem1_; // First adjacent element
        std::vector<Index> faceElem2_; // Second adjacent element (InvalidIndex for boundary)
        std::vector<int> faceLocal1_; // Local face index in elem1
        std::vector<int> faceLocal2_; // Local face index in elem2
        std::vector<char> faceBoundary_; // char (0=interior, 1=boundary)

        // Boundary face tracking
        std::vector<Index> boundaryFaceIndices_;
        std::vector<Index> interiorFaceIndices_;

        // Boundary element to face mapping
        std::unordered_map<Index, Index> bdrElementToFace_;
        std::unordered_map<Index, bool> bdrIdExternalCache_; ///< Cache: boundary ID -> isExternal
    };

} // namespace mpfem

#endif // MPFEM_MESH_HPP
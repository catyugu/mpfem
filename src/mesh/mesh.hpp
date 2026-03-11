#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "geometry.hpp"
#include "vertex.hpp"
#include "element.hpp"
#include "core/types.hpp"
#include "core/exception.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <span>

namespace mpfem {

// Forward declaration
class MeshTopology;

/**
 * @brief Mesh class storing vertices and elements.
 * 
 * This class manages:
 * - Vertex coordinates
 * - Volume elements (tetrahedra, hexahedra, etc.)
 * - Boundary elements (triangles, quads)
 * - Domain and boundary attributes
 * 
 * The mesh class does NOT depend on any external library.
 */
class Mesh {
public:
    /// Default constructor
    Mesh() = default;

    /// Construct with pre-allocated sizes
    Mesh(int dim, Index numVertices, Index numElements, Index numBdrElements = 0)
        : dim_(dim)
        , vertices_(numVertices)
        , elements_(numElements)
        , bdrElements_(numBdrElements) {}

    // -------------------------------------------------------------------------
    // Dimension and size
    // -------------------------------------------------------------------------

    /// Get spatial dimension
    int dim() const { return dim_; }

    /// Set spatial dimension
    void setDim(int dim) { dim_ = dim; }

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

    /// Get vertex coordinates as Eigen vector
    Vector3 vertexCoords(Index i) const { return vertices_[i].toVector(); }

    /// Add a vertex
    Index addVertex(const Vertex& v) {
        vertices_.push_back(v);
        return static_cast<Index>(vertices_.size() - 1);
    }

    /// Add a vertex from coordinates
    Index addVertex(Real x, Real y = 0.0, Real z = 0.0) {
        vertices_.emplace_back(x, y, z, dim_);
        return static_cast<Index>(vertices_.size() - 1);
    }

    /// Set vertices (for bulk loading)
    void setVertices(std::vector<Vertex>&& vertices) {
        vertices_ = std::move(vertices);
    }

    /// Reserve space for vertices
    void reserveVertices(Index n) { vertices_.reserve(n); }

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
    Index addElement(const Element& e) {
        elements_.push_back(e);
        return static_cast<Index>(elements_.size() - 1);
    }

    /// Add an element from geometry and vertices
    Index addElement(Geometry geom, std::vector<Index> vertices, Index attribute = 0) {
        elements_.emplace_back(geom, std::move(vertices), attribute);
        return static_cast<Index>(elements_.size() - 1);
    }

    /// Set elements (for bulk loading)
    void setElements(std::vector<Element>&& elements) {
        elements_ = std::move(elements);
    }

    /// Reserve space for elements
    void reserveElements(Index n) { elements_.reserve(n); }

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
    Index addBdrElement(const Element& e) {
        bdrElements_.push_back(e);
        return static_cast<Index>(bdrElements_.size() - 1);
    }

    /// Add a boundary element from geometry and vertices
    Index addBdrElement(Geometry geom, std::vector<Index> vertices, Index attribute = 0) {
        bdrElements_.emplace_back(geom, std::move(vertices), attribute);
        return static_cast<Index>(bdrElements_.size() - 1);
    }

    /// Set boundary elements (for bulk loading)
    void setBdrElements(std::vector<Element>&& elements) {
        bdrElements_ = std::move(elements);
    }

    /// Reserve space for boundary elements
    void reserveBdrElements(Index n) { bdrElements_.reserve(n); }

    // -------------------------------------------------------------------------
    // Domain and boundary attribute access
    // -------------------------------------------------------------------------

    /// Get all unique domain IDs
    std::vector<Index> getDomainIds() const {
        std::vector<Index> ids;
        for (const auto& elem : elements_) {
            ids.push_back(elem.attribute());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }

    /// Get all unique boundary IDs
    std::vector<Index> getBoundaryIds() const {
        std::vector<Index> ids;
        for (const auto& elem : bdrElements_) {
            ids.push_back(elem.attribute());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }

    /// Get elements in a specific domain
    std::vector<Index> getElementsInDomain(Index domainId) const {
        std::vector<Index> elemIds;
        for (Index i = 0; i < static_cast<Index>(elements_.size()); ++i) {
            if (elements_[i].attribute() == domainId) {
                elemIds.push_back(i);
            }
        }
        return elemIds;
    }

    /// Get boundary elements with a specific boundary ID
    std::vector<Index> getBdrElementsWithId(Index boundaryId) const {
        std::vector<Index> elemIds;
        for (Index i = 0; i < static_cast<Index>(bdrElements_.size()); ++i) {
            if (bdrElements_[i].attribute() == boundaryId) {
                elemIds.push_back(i);
            }
        }
        return elemIds;
    }

    // -------------------------------------------------------------------------
    // Geometry queries
    // -------------------------------------------------------------------------

    /// Check if mesh contains a specific geometry type
    bool hasGeometry(Geometry geom) const {
        for (const auto& elem : elements_) {
            if (elem.geometry() == geom) return true;
        }
        return false;
    }

    /// Get element geometry types present in the mesh
    std::vector<Geometry> getGeometryTypes() const {
        std::vector<Geometry> types;
        for (const auto& elem : elements_) {
            if (std::find(types.begin(), types.end(), elem.geometry()) == types.end()) {
                types.push_back(elem.geometry());
            }
        }
        return types;
    }

    // -------------------------------------------------------------------------
    // Element geometry
    // -------------------------------------------------------------------------

    /// Get vertex coordinates for an element
    std::vector<Vector3> getElementVertices(Index elemIdx) const {
        std::vector<Vector3> coords;
        const auto& elem = elements_[elemIdx];
        coords.reserve(elem.numVertices());
        for (int i = 0; i < elem.numVertices(); ++i) {
            coords.push_back(vertices_[elem.vertex(i)].toVector());
        }
        return coords;
    }

    /// Get vertex coordinates for a boundary element
    std::vector<Vector3> getBdrElementVertices(Index bdrElemIdx) const {
        std::vector<Vector3> coords;
        const auto& elem = bdrElements_[bdrElemIdx];
        coords.reserve(elem.numVertices());
        for (int i = 0; i < elem.numVertices(); ++i) {
            coords.push_back(vertices_[elem.vertex(i)].toVector());
        }
        return coords;
    }

    // -------------------------------------------------------------------------
    // Topology
    // -------------------------------------------------------------------------

    /// Get the topology object (const)
    const MeshTopology& topology() const;

    /// Build topology (element-face, face-element, etc.)
    void buildTopology() const;

    /// Check if topology has been built
    bool hasTopology() const { return topology_ != nullptr; }

    // -------------------------------------------------------------------------
    // Statistics and validation
    // -------------------------------------------------------------------------

    /// Print mesh statistics
    void printStats() const;

    /// Validate mesh
    bool validate() const;

    /// Get bounding box
    std::pair<Vector3, Vector3> getBoundingBox() const;

private:
    int dim_ = 3;  // Spatial dimension
    std::vector<Vertex> vertices_;
    std::vector<Element> elements_;      // Volume elements
    std::vector<Element> bdrElements_;   // Boundary elements (codim-1)
    
    mutable std::unique_ptr<MeshTopology> topology_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_HPP

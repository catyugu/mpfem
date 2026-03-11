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

namespace mpfem {

// Forward declaration
class MeshTopology;

/**
 * @brief Mesh class storing vertices and elements.
 * 
 * This class manages:
 * - Vertex coordinates
 * - Volume elements (tetrahedra, hexahedra)
 * - Boundary elements (triangles, quads)
 * - Domain and boundary attributes
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
    Index addElement(Geometry geom, std::span<const Index> vertices, Index attr = 0);
    Index addElement(Geometry geom, const std::vector<Index>& vertices, Index attr = 0);

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
    Index addBdrElement(Geometry geom, std::span<const Index> vertices, Index attr = 0);
    Index addBdrElement(Geometry geom, const std::vector<Index>& vertices, Index attr = 0);

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
    // Utility
    // -------------------------------------------------------------------------

    /// Clear all data
    void clear();

    /// Get bounding box (min, max)
    std::pair<Vector3, Vector3> getBoundingBox() const;

private:
    int dim_ = 3;
    std::vector<Vertex> vertices_;
    std::vector<Element> elements_;
    std::vector<Element> bdrElements_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_HPP
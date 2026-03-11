#ifndef MPFEM_ELEMENT_HPP
#define MPFEM_ELEMENT_HPP

#include "geometry.hpp"
#include "core/types.hpp"
#include <vector>
#include <span>

namespace mpfem {

/**
 * @brief Element class representing a mesh element (topology only).
 * 
 * An element stores:
 * - Geometry type (shape)
 * - Vertex indices (connectivity)
 * - Element order (1 = linear, 2 = quadratic)
 * - Domain/boundary attribute
 * 
 * This class is purely topological - it does not store coordinates.
 * Coordinate data is managed by the Mesh class.
 * 
 * For second-order elements, the vertex ordering follows COMSOL convention:
 * - First N corner nodes (N = numCorners(geometry))
 * - Followed by edge midpoint nodes
 * - Followed by face nodes (for hex elements)
 * - Followed by interior node (for hex elements)
 * 
 * Note: Edge and face topology is delegated to geom namespace functions,
 * which operate on the reference element topology. This keeps the Element
 * class simple and decouples topology from geometry.
 */
class Element {
public:
    /// Default constructor
    Element() = default;

    /// Construct from geometry type and vertex indices (assumes first order)
    Element(Geometry geom, std::vector<Index> vertices, Index attribute = 0)
        : geometry_(geom)
        , vertices_(std::move(vertices))
        , attribute_(attribute)
        , order_(1) {
        validate();
    }

    /// Construct from geometry type, vertex indices, and order
    Element(Geometry geom, std::vector<Index> vertices, Index attribute, int order)
        : geometry_(geom)
        , vertices_(std::move(vertices))
        , attribute_(attribute)
        , order_(order) {
        validate();
    }

    /// Construct from geometry type and span (assumes first order)
    Element(Geometry geom, std::span<const Index> vertices, Index attribute = 0)
        : geometry_(geom)
        , vertices_(vertices.begin(), vertices.end())
        , attribute_(attribute)
        , order_(1) {
        validate();
    }

    /// Construct from geometry type, span, and order
    Element(Geometry geom, std::span<const Index> vertices, Index attribute, int order)
        : geometry_(geom)
        , vertices_(vertices.begin(), vertices.end())
        , attribute_(attribute)
        , order_(order) {
        validate();
    }

    // -------------------------------------------------------------------------
    // Geometry access (delegated to geom namespace)
    // -------------------------------------------------------------------------

    /// Get geometry type
    Geometry geometry() const { return geometry_; }

    /// Get spatial dimension of the element
    int dim() const { return geom::dim(geometry_); }

    /// Get number of vertices
    int numVertices() const { return static_cast<int>(vertices_.size()); }

    /// Get number of edges (from geometry)
    int numEdges() const { return geom::numEdges(geometry_); }

    /// Get number of faces (from geometry)
    int numFaces() const { return geom::numFaces(geometry_); }

    /// Check if element is a volume element
    bool isVolume() const { return geom::isVolume(geometry_); }

    /// Check if element is a surface element
    bool isSurface() const { return geom::isSurface(geometry_); }

    // -------------------------------------------------------------------------
    // Vertex access
    // -------------------------------------------------------------------------

    /// Get vertex index
    Index vertex(int i) const { return vertices_[i]; }
    Index& vertex(int i) { return vertices_[i]; }

    /// Get all vertex indices
    const std::vector<Index>& vertices() const { return vertices_; }
    std::vector<Index>& vertices() { return vertices_; }

    /// Get vertex indices as span
    std::span<const Index> vertexSpan() const { return vertices_; }

    // -------------------------------------------------------------------------
    // Attribute access
    // -------------------------------------------------------------------------

    /// Get attribute (domain ID for volume elements, boundary ID for surface elements)
    Index attribute() const { return attribute_; }
    Index& attribute() { return attribute_; }

    /// Set attribute
    void setAttribute(Index attr) { attribute_ = attr; }

    // -------------------------------------------------------------------------
    // Order access
    // -------------------------------------------------------------------------

    /// Get element order (1 = linear, 2 = quadratic)
    int order() const { return order_; }

    /// Set element order
    void setOrder(int order) { order_ = order; }

    /// Check if element is quadratic (second-order)
    bool isQuadratic() const { return order_ >= 2; }

    /// Get number of corner vertices (first-order nodes)
    int numCorners() const { return geom::numCorners(geometry_); }

    // -------------------------------------------------------------------------
    // Edge and face connectivity (using geom namespace)
    // -------------------------------------------------------------------------

    /**
     * @brief Get global vertex indices for an edge.
     * @param edgeIdx Edge index (0 to numEdges()-1)
     * @return Pair of global vertex indices
     */
    std::pair<Index, Index> edgeVertices(int edgeIdx) const {
        auto local = geom::edgeVertices(geometry_, edgeIdx);
        return {vertices_[local.first], vertices_[local.second]};
    }

    /**
     * @brief Get global vertex indices for a face.
     * @param faceIdx Face index (0 to numFaces()-1)
     * @return Vector of global vertex indices for the face
     */
    std::vector<Index> faceVertices(int faceIdx) const {
        std::vector<Index> result;
        auto localVerts = geom::faceVertices(geometry_, faceIdx);
        result.reserve(localVerts.size());
        for (int lv : localVerts) {
            result.push_back(vertices_[lv]);
        }
        return result;
    }

    /**
     * @brief Get the geometry type of a face.
     * @param faceIdx Face index
     * @return Geometry type of the face
     */
    Geometry faceGeometry(int faceIdx) const {
        return geom::faceGeometry(geometry_, faceIdx);
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    /// Check if the element is valid
    bool isValid() const {
        if (geometry_ == Geometry::Invalid) return false;
        int expectedVerts = geom::numVertices(geometry_, order_);
        return static_cast<int>(vertices_.size()) == expectedVerts;
    }

    /// Get human-readable name
    std::string_view name() const { return geom::name(geometry_); }

private:
    void validate() {
        // Check vertex count matches geometry and order
        int expectedVerts = geom::numVertices(geometry_, order_);
        if (static_cast<int>(vertices_.size()) != expectedVerts) {
            // Could throw an exception, but for now just mark as invalid
            geometry_ = Geometry::Invalid;
        }
    }

    Geometry geometry_ = Geometry::Invalid;
    std::vector<Index> vertices_;
    Index attribute_ = 0;  // Domain ID or boundary ID
    int order_ = 1;        // Element order (1 = linear, 2 = quadratic)
};

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_HPP

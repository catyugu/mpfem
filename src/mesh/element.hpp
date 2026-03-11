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
    // Geometry access
    // -------------------------------------------------------------------------

    /// Get geometry type
    Geometry geometry() const { return geometry_; }

    /// Get spatial dimension of the element
    int dim() const { return geom::dim(geometry_); }

    /// Get number of vertices
    int numVertices() const { return static_cast<int>(vertices_.size()); }

    /// Get number of edges
    int numEdges() const { return geom::numEdges(geometry_); }

    /// Get number of faces
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
    // Edge and face connectivity (computed on demand)
    // -------------------------------------------------------------------------

    /**
     * @brief Get vertex indices for an edge.
     * @param edgeIdx Edge index (0 to numEdges()-1)
     * @return Pair of vertex indices
     */
    std::pair<Index, Index> edgeVertices(int edgeIdx) const;

    /**
     * @brief Get vertex indices for a face.
     * @param faceIdx Face index (0 to numFaces()-1)
     * @return Vector of vertex indices for the face
     */
    std::vector<Index> faceVertices(int faceIdx) const;

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

// =============================================================================
// Edge vertex tables (local vertex indices for each edge)
// =============================================================================

namespace edge_table {

/// Edge vertices for Triangle: 3 edges
inline constexpr std::array<std::pair<int, int>, 3> Triangle = {{
    {1, 2}, {2, 0}, {0, 1}
}};

/// Edge vertices for Square: 4 edges
inline constexpr std::array<std::pair<int, int>, 4> Square = {{
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
}};

/// Edge vertices for Tetrahedron: 6 edges
inline constexpr std::array<std::pair<int, int>, 6> Tetrahedron = {{
    {0, 1}, {1, 2}, {2, 0},  // Bottom face edges
    {0, 3}, {1, 3}, {2, 3}   // Side edges
}};

/// Edge vertices for Cube: 12 edges
inline constexpr std::array<std::pair<int, int>, 12> Cube = {{
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
}};

}  // namespace edge_table

// =============================================================================
// Face vertex tables (local vertex indices for each face)
// =============================================================================

namespace face_table {

/// Face vertices for Tetrahedron: 4 triangular faces
inline constexpr std::array<std::array<int, 3>, 4> Tetrahedron = {{
    {{1, 2, 3}},  // Face opposite vertex 0
    {{0, 3, 2}},  // Face opposite vertex 1 (reversed for outward normal)
    {{0, 1, 3}},  // Face opposite vertex 2
    {{0, 2, 1}}   // Face opposite vertex 3 (reversed for outward normal)
}};

/// Face vertices for Cube: 6 quadrilateral faces
/// Ordering: -z, +z, -x, +x, -y, +y (for outward normals)
inline constexpr std::array<std::array<int, 4>, 6> Cube = {{
    {{0, 1, 2, 3}},  // Bottom (-z)
    {{4, 7, 6, 5}},  // Top (+z)
    {{0, 3, 7, 4}},  // Front (-x)
    {{1, 5, 6, 2}},  // Back (+x)
    {{0, 4, 5, 1}},  // Left (-y)
    {{3, 2, 6, 7}}   // Right (+y)
}};

}  // namespace face_table

// =============================================================================
// Element inline methods
// =============================================================================

inline std::pair<Index, Index> Element::edgeVertices(int edgeIdx) const {
    const std::pair<int, int>* table = nullptr;
    int numEdges = 0;
    
    switch (geometry_) {
        case Geometry::Triangle:
            table = reinterpret_cast<const std::pair<int, int>*>(edge_table::Triangle.data());
            numEdges = 3;
            break;
        case Geometry::Square:
            table = reinterpret_cast<const std::pair<int, int>*>(edge_table::Square.data());
            numEdges = 4;
            break;
        case Geometry::Tetrahedron:
            table = reinterpret_cast<const std::pair<int, int>*>(edge_table::Tetrahedron.data());
            numEdges = 6;
            break;
        case Geometry::Cube:
            table = reinterpret_cast<const std::pair<int, int>*>(edge_table::Cube.data());
            numEdges = 12;
            break;
        default:
            return {0, 0};
    }
    
    if (edgeIdx < 0 || edgeIdx >= numEdges) {
        return {0, 0};
    }
    
    return {vertices_[table[edgeIdx].first], vertices_[table[edgeIdx].second]};
}

inline std::vector<Index> Element::faceVertices(int faceIdx) const {
    std::vector<Index> result;
    
    switch (geometry_) {
        case Geometry::Tetrahedron: {
            if (faceIdx >= 0 && faceIdx < 4) {
                result = {vertices_[face_table::Tetrahedron[faceIdx][0]],
                          vertices_[face_table::Tetrahedron[faceIdx][1]],
                          vertices_[face_table::Tetrahedron[faceIdx][2]]};
            }
            break;
        }
        case Geometry::Cube: {
            if (faceIdx >= 0 && faceIdx < 6) {
                result = {vertices_[face_table::Cube[faceIdx][0]],
                          vertices_[face_table::Cube[faceIdx][1]],
                          vertices_[face_table::Cube[faceIdx][2]],
                          vertices_[face_table::Cube[faceIdx][3]]};
            }
            break;
        }
        case Geometry::Triangle:
        case Geometry::Square: {
            // 2D elements: the "face" is the element itself
            result = vertices_;
            break;
        }
        default:
            break;
    }
    
    return result;
}

}  // namespace mpfem

#endif  // MPFEM_ELEMENT_HPP
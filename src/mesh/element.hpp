/**
 * @file element.hpp
 * @brief Element type definitions for mpfem
 */

#ifndef MPFEM_MESH_ELEMENT_HPP
#define MPFEM_MESH_ELEMENT_HPP

#include "core/types.hpp"

#include <array>
#include <span>
#include <vector>

namespace mpfem {

// ============================================================
// Element Type Enumeration
// ============================================================

/// Element type codes (compatible with COMSOL/VTK conventions)
enum class ElementType : int {
    Vertex = 0,        ///< Point element
    Segment = 1,       ///< Line segment
    Triangle = 2,      ///< Triangle (2D)
    Quadrilateral = 3, ///< Quadrilateral (2D)
    Tetrahedron = 4,   ///< Tetrahedron (3D)
    Hexahedron = 5,    ///< Hexahedron (3D)
    Wedge = 6,         ///< Prism/Wedge (3D)
    Pyramid = 7,       ///< Pyramid (3D)
    // Quadratic elements (second order)
    Segment2 = 8,      ///< Quadratic segment
    Triangle2 = 9,     ///< Quadratic triangle
    Quadrilateral2 = 10, ///< Quadratic quadrilateral
    Tetrahedron2 = 11, ///< Quadratic tetrahedron
    Hexahedron2 = 12,  ///< Quadratic hexahedron
    Wedge2 = 13,       ///< Quadratic wedge
    Pyramid2 = 14,     ///< Quadratic pyramid
    Unknown = -1
};

// ============================================================
// Element Trait Templates
// ============================================================

namespace detail {

/// Number of vertices for each element type (linear)
constexpr int num_vertices_v[] = {
    1,   // Vertex
    2,   // Segment
    3,   // Triangle
    4,   // Quadrilateral
    4,   // Tetrahedron
    8,   // Hexahedron
    6,   // Wedge
    5,   // Pyramid
    // Quadratic
    3,   // Segment2
    6,   // Triangle2
    8,   // Quadrilateral2
    10,  // Tetrahedron2
    20,  // Hexahedron2
    15,  // Wedge2
    13   // Pyramid2
};

/// Number of nodes (DoFs) for each element type
constexpr int num_nodes_v[] = {
    1,   // Vertex
    2,   // Segment
    3,   // Triangle
    4,   // Quadrilateral
    4,   // Tetrahedron
    8,   // Hexahedron
    6,   // Wedge
    5,   // Pyramid
    // Quadratic
    3,   // Segment2
    6,   // Triangle2
    8,   // Quadrilateral2
    10,  // Tetrahedron2
    20,  // Hexahedron2
    15,  // Wedge2
    13   // Pyramid2
};

/// Dimension for each element type
constexpr int dimension_v[] = {
    0,   // Vertex
    1,   // Segment
    2,   // Triangle
    2,   // Quadrilateral
    3,   // Tetrahedron
    3,   // Hexahedron
    3,   // Wedge
    3,   // Pyramid
    // Quadratic
    1,   // Segment2
    2,   // Triangle2
    2,   // Quadrilateral2
    3,   // Tetrahedron2
    3,   // Hexahedron2
    3,   // Wedge2
    3    // Pyramid2
};

/// Geometry type mapping
constexpr GeometryType to_geometry_v[] = {
    GeometryType::Point,         // Vertex
    GeometryType::Segment,       // Segment
    GeometryType::Triangle,      // Triangle
    GeometryType::Quadrilateral, // Quadrilateral
    GeometryType::Tetrahedron,   // Tetrahedron
    GeometryType::Hexahedron,    // Hexahedron
    GeometryType::Wedge,         // Wedge
    GeometryType::Pyramid,       // Pyramid
    // Quadratic (same geometry type)
    GeometryType::Segment,       // Segment2
    GeometryType::Triangle,      // Triangle2
    GeometryType::Quadrilateral, // Quadrilateral2
    GeometryType::Tetrahedron,   // Tetrahedron2
    GeometryType::Hexahedron,    // Hexahedron2
    GeometryType::Wedge,         // Wedge2
    GeometryType::Pyramid        // Pyramid2
};

}  // namespace detail

// ============================================================
// Element Type Utilities
// ============================================================

/// Get number of vertices for element type
constexpr int num_vertices(ElementType type) {
    const int idx = static_cast<int>(type);
    return (idx >= 0 && idx < 15) ? detail::num_vertices_v[idx] : 0;
}

/// Get number of nodes for element type
constexpr int num_nodes(ElementType type) { return num_vertices(type); }

/// Get dimension of element type
constexpr int element_dimension(ElementType type) {
    const int idx = static_cast<int>(type);
    return (idx >= 0 && idx < 15) ? detail::dimension_v[idx] : -1;
}

/// Convert element type to geometry type
constexpr GeometryType to_geometry_type(ElementType type) {
    const int idx = static_cast<int>(type);
    return (idx >= 0 && idx < 15) ? detail::to_geometry_v[idx]
                                 : GeometryType::Point;
}

/// Check if element type is quadratic (second order)
constexpr bool is_quadratic(ElementType type) {
    const int idx = static_cast<int>(type);
    return idx >= 8 && idx < 15;
}

/// Get element order (1 for linear, 2 for quadratic)
constexpr int element_order(ElementType type) {
    return is_quadratic(type) ? 2 : 1;
}

/// Get element type name as string
inline const char* element_type_name(ElementType type) {
    switch (type) {
        case ElementType::Vertex:
            return "Vertex";
        case ElementType::Segment:
            return "Segment";
        case ElementType::Triangle:
            return "Triangle";
        case ElementType::Quadrilateral:
            return "Quadrilateral";
        case ElementType::Tetrahedron:
            return "Tetrahedron";
        case ElementType::Hexahedron:
            return "Hexahedron";
        case ElementType::Wedge:
            return "Wedge";
        case ElementType::Pyramid:
            return "Pyramid";
        case ElementType::Segment2:
            return "Segment2";
        case ElementType::Triangle2:
            return "Triangle2";
        case ElementType::Quadrilateral2:
            return "Quadrilateral2";
        case ElementType::Tetrahedron2:
            return "Tetrahedron2";
        case ElementType::Hexahedron2:
            return "Hexahedron2";
        case ElementType::Wedge2:
            return "Wedge2";
        case ElementType::Pyramid2:
            return "Pyramid2";
        default:
            return "Unknown";
    }
}

/// Parse element type from string (COMSOL format)
inline ElementType parse_element_type(const std::string& name) {
    if (name == "vtx")
        return ElementType::Vertex;
    if (name == "edg")
        return ElementType::Segment;
    if (name == "tri")
        return ElementType::Triangle;
    if (name == "quad")
        return ElementType::Quadrilateral;
    if (name == "tet")
        return ElementType::Tetrahedron;
    if (name == "hex")
        return ElementType::Hexahedron;
    if (name == "prism" || name == "wedge")
        return ElementType::Wedge;
    if (name == "pyr")
        return ElementType::Pyramid;
    // Quadratic
    if (name == "edg2" || name == "seg2")
        return ElementType::Segment2;
    if (name == "tri2")
        return ElementType::Triangle2;
    if (name == "quad2")
        return ElementType::Quadrilateral2;
    if (name == "tet2")
        return ElementType::Tetrahedron2;
    if (name == "hex2")
        return ElementType::Hexahedron2;
    if (name == "prism2" || name == "wedge2")
        return ElementType::Wedge2;
    if (name == "pyr2")
        return ElementType::Pyramid2;
    return ElementType::Unknown;
}

// ============================================================
// Element Class
// ============================================================

/**
 * @brief Represents a single mesh element
 * 
 * Stores the element type and vertex indices.
 * Supports both linear and quadratic elements.
 */
class Element {
public:
    Element() : type_(ElementType::Unknown), entity_id_(0) {}

    Element(ElementType type, std::vector<Index> vertices, Index entity_id = 0)
        : type_(type), vertices_(std::move(vertices)), entity_id_(entity_id) {}

    /// Get element type
    ElementType type() const { return type_; }

    /// Get geometry type
    GeometryType geometry_type() const {
        return to_geometry_type(type_);
    }

    /// Get element dimension
    int dimension() const { return element_dimension(type_); }

    /// Get element order (1 for linear, 2 for quadratic)
    int order() const { return element_order(type_); }

    /// Get number of vertices/nodes
    int num_vertices() const { return static_cast<int>(vertices_.size()); }

    /// Get vertex index
    Index vertex(int i) const { return vertices_[i]; }

    /// Get all vertices
    std::span<const Index> vertices() const { return vertices_; }

    /// Get mutable vertices
    std::vector<Index>& vertices() { return vertices_; }

    /// Get geometric entity ID (domain or boundary ID)
    Index entity_id() const { return entity_id_; }

    /// Set geometric entity ID
    void set_entity_id(Index id) { entity_id_ = id; }

    /// Check if element is valid
    bool is_valid() const {
        return type_ != ElementType::Unknown &&
               static_cast<int>(vertices_.size()) == num_nodes(type_);
    }

private:
    ElementType type_;
    std::vector<Index> vertices_;
    Index entity_id_;  ///< Domain ID (for volume) or Boundary ID (for surface)
};

// ============================================================
// Element Block (Collection of same-type elements)
// ============================================================

/**
 * @brief A block of elements of the same type
 * 
 * Used for efficient storage and processing of mesh elements.
 * Elements are stored in a flat array with fixed stride.
 */
class ElementBlock {
public:
    ElementBlock() : type_(ElementType::Unknown), nodes_per_element_(0) {}

    ElementBlock(ElementType type) : type_(type), nodes_per_element_(num_nodes(type)) {}

    /// Get element type
    ElementType type() const { return type_; }

    /// Get number of elements in block
    SizeType size() const { return entity_ids_.size(); }

    /// Get nodes per element
    int nodes_per_element() const { return nodes_per_element_; }

    /// Add an element
    void add_element(std::span<const Index> vertices, Index entity_id) {
        if (vertices.size() != static_cast<size_t>(nodes_per_element_)) {
            return;  // Invalid element
        }
        for (auto v : vertices) {
            connectivity_.push_back(v);
        }
        entity_ids_.push_back(entity_id);
    }

    /// Get vertex indices for element i
    std::span<const Index> element_vertices(SizeType i) const {
        const Index offset = static_cast<Index>(i * nodes_per_element_);
        return std::span<const Index>(connectivity_.data() + offset,
                                      nodes_per_element_);
    }

    /// Get entity ID for element i
    Index entity_id(SizeType i) const { return entity_ids_[i]; }

    /// Get all connectivity data
    const IndexArray& connectivity() const { return connectivity_; }

    /// Get all entity IDs
    const IndexArray& entity_ids() const { return entity_ids_; }

    /// Get mutable entity IDs (for setting during parsing)
    IndexArray& entity_ids_mut() { return entity_ids_; }

    /// Set entity ID for element i
    void set_entity_id(SizeType i, Index entity_id) {
        if (i < entity_ids_.size()) {
            entity_ids_[i] = entity_id;
        }
    }

    /// Check if block is empty
    bool empty() const { return entity_ids_.empty(); }

    /// Clear all data
    void clear() {
        connectivity_.clear();
        entity_ids_.clear();
    }

    /// Reserve capacity
    void reserve(SizeType num_elements) {
        connectivity_.reserve(num_elements * nodes_per_element_);
        entity_ids_.reserve(num_elements);
    }

private:
    ElementType type_;
    int nodes_per_element_;
    IndexArray connectivity_;  // Flat array of vertex indices
    IndexArray entity_ids_;    // One per element
};

}  // namespace mpfem

#endif  // MPFEM_MESH_ELEMENT_HPP

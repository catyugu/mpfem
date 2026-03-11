#ifndef MPFEM_GEOMETRY_HPP
#define MPFEM_GEOMETRY_HPP

#include <cstdint>
#include <string_view>
#include <array>
#include <utility>
#include <vector>

namespace mpfem
{

    /**
     * @brief Geometry type enumeration for different element shapes.
     *
     * This enum defines the supported geometric element types.
     * The values are designed to be compatible with MFEM's Geometry::Type.
     *
     * Supported types:
     * - Point, Segment: 0D and 1D
     * - Triangle, Square: 2D
     * - Tetrahedron, Cube: 3D
     */
    enum class Geometry : std::uint8_t
    {
        Point = 0,       ///< 0D: Point
        Segment = 1,     ///< 1D: Line segment (2 vertices)
        Triangle = 2,    ///< 2D: Triangle (3 vertices)
        Square = 3,      ///< 2D: Quadrilateral (4 vertices)
        Tetrahedron = 4, ///< 3D: Tetrahedron (4 vertices)
        Cube = 5,        ///< 3D: Hexahedron (8 vertices)

        // Aliases for convenience
        Quad = Square,
        Hex = Cube,
        Tet = Tetrahedron,

        // Sentinel
        Invalid = 255
    };

    // =============================================================================
    // Geometry traits
    // =============================================================================

    namespace geom
    {

        /// Get the dimension of a geometry type
        constexpr int dim(Geometry g)
        {
            switch (g)
            {
            case Geometry::Point:
                return 0;
            case Geometry::Segment:
                return 1;
            case Geometry::Triangle:
            case Geometry::Square:
                return 2;
            case Geometry::Tetrahedron:
            case Geometry::Cube:
                return 3;
            default:
                return -1;
            }
        }

        /// Get the number of vertices for a geometry type (first order)
        constexpr int numVertices(Geometry g)
        {
            switch (g)
            {
            case Geometry::Point:
                return 1;
            case Geometry::Segment:
                return 2;
            case Geometry::Triangle:
                return 3;
            case Geometry::Square:
                return 4;
            case Geometry::Tetrahedron:
                return 4;
            case Geometry::Cube:
                return 8;
            default:
                return 0;
            }
        }

        /// Get the number of vertices for a geometry type with given order
        /// @param g Geometry type
        /// @param order Element order (1 = linear, 2 = quadratic)
        constexpr int numVertices(Geometry g, int order)
        {
            if (order <= 1)
                return numVertices(g);

            // Second order elements have additional edge midpoints
            switch (g)
            {
            case Geometry::Point:
                return 1;
            case Geometry::Segment:
                return 2 + 1; // 2 corner + 1 edge midpoint
            case Geometry::Triangle:
                return 3 + 3; // 3 corner + 3 edge midpoints
            case Geometry::Square:
                return 4 + 4 + 1; // 4 corner + 4 edge midpoints + 1 center
            case Geometry::Tetrahedron:
                return 4 + 6; // 4 corner + 6 edge midpoints
            case Geometry::Cube:
                return 8 + 12 + 6 + 1; // 8 corner + 12 edges + 6 faces + 1 center
            default:
                return 0;
            }
        }

        /// Get the number of edges for a geometry type

        constexpr int numEdges(Geometry g)
        {

            switch (g)
            {

            case Geometry::Point:
                return 0;

            case Geometry::Segment:
                return 1;

            case Geometry::Triangle:
                return 3;

            case Geometry::Square:
                return 4;

            case Geometry::Tetrahedron:
                return 6;

            case Geometry::Cube:
                return 12;

            default:
                return 0;
            }
        }

        /// Get the number of corner vertices (first-order nodes) for a geometry type

        constexpr int numCorners(Geometry g)
        {

            return numVertices(g); // Same as first-order vertices
        }

        /// Get the number of edge midpoints for a second-order element

        constexpr int numEdgeMidpoints(Geometry g)
        {

            return numEdges(g); // One midpoint per edge
        }

        /// Get the number of face nodes for a second-order element (center of face for quad)

        constexpr int numFaceNodes(Geometry g)
        {

            switch (g)
            {

            case Geometry::Square:
                return 1; // Center node

            case Geometry::Cube:
                return 6; // One center per face

            default:
                return 0;
            }
        }

        /// Get the number of interior nodes for a second-order element

        constexpr int numInteriorNodes(Geometry g)
        {

            switch (g)
            {

            case Geometry::Cube:
                return 1; // Center node

            default:
                return 0;
            }
        }

        /// Get the number of faces for a geometry type
        constexpr int numFaces(Geometry g)
        {
            switch (g)
            {
            case Geometry::Point:
            case Geometry::Segment:
                return 0;
            case Geometry::Triangle:
            case Geometry::Square:
                return 1; // Self is the face
            case Geometry::Tetrahedron:
                return 4;
            case Geometry::Cube:
                return 6;
            default:
                return 0;
            }
        }

        /// Get the geometry type of faces for a 3D element
        constexpr Geometry faceGeometry(Geometry g, int /*faceIdx*/)
        {
            switch (g)
            {
            case Geometry::Tetrahedron:
                return Geometry::Triangle;
            case Geometry::Cube:
                return Geometry::Square;
            default:
                return Geometry::Invalid;
            }
        }

        /// Get the geometry type of edges
        constexpr Geometry edgeGeometry(Geometry /*g*/)
        {
            return Geometry::Segment;
        }

        /// Get human-readable name for geometry type
        constexpr std::string_view name(Geometry g)
        {
            switch (g)
            {
            case Geometry::Point:
                return "Point";
            case Geometry::Segment:
                return "Segment";
            case Geometry::Triangle:
                return "Triangle";
            case Geometry::Square:
                return "Square";
            case Geometry::Tetrahedron:
                return "Tetrahedron";
            case Geometry::Cube:
                return "Cube";
            default:
                return "Invalid";
            }
        }

        /// Check if geometry is a simplex (triangle, tetrahedron)
        constexpr bool isSimplex(Geometry g)
        {
            return g == Geometry::Triangle || g == Geometry::Tetrahedron;
        }

        /// Check if geometry is a tensor product element (square, cube)
        constexpr bool isTensorProduct(Geometry g)
        {
            return g == Geometry::Segment || g == Geometry::Square || g == Geometry::Cube;
        }

        /// Check if geometry is a volume element
        constexpr bool isVolume(Geometry g)
        {
            return g == Geometry::Tetrahedron || g == Geometry::Cube;
        }

        /// Check if geometry is a surface element
        constexpr bool isSurface(Geometry g)
        {
            return g == Geometry::Triangle || g == Geometry::Square;
        }

        // =============================================================================
        // Reference element vertex coordinates
        // =============================================================================

        /// Reference coordinates for each geometry type
        /// Note: These are defined for the standard reference elements

        /// Reference vertex coordinates for Segment: [-1, 1]
        inline constexpr std::array<std::array<double, 1>, 2> refCoords_Segment = {{{{-1.0}}, {{1.0}}}};

        /// Reference vertex coordinates for Triangle: (0,0), (1,0), (0,1)
        inline constexpr std::array<std::array<double, 2>, 3> refCoords_Triangle = {{{{0.0, 0.0}}, {{1.0, 0.0}}, {{0.0, 1.0}}}};

        /// Reference vertex coordinates for Square: [-1,1] x [-1,1]
        inline constexpr std::array<std::array<double, 2>, 4> refCoords_Square = {{{{-1.0, -1.0}}, {{1.0, -1.0}}, {{1.0, 1.0}}, {{-1.0, 1.0}}}};

        /// Reference vertex coordinates for Tetrahedron
        inline constexpr std::array<std::array<double, 3>, 4> refCoords_Tetrahedron = {{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};

        /// Reference vertex coordinates for Cube: [-1,1]^3
        inline constexpr std::array<std::array<double, 3>, 8> refCoords_Cube = {{{{-1.0, -1.0, -1.0}}, {{1.0, -1.0, -1.0}}, {{1.0, 1.0, -1.0}}, {{-1.0, 1.0, -1.0}}, {{-1.0, -1.0, 1.0}}, {{1.0, -1.0, 1.0}}, {{1.0, 1.0, 1.0}}, {{-1.0, 1.0, 1.0}}}};

        // =============================================================================
        // Reference element topology tables
        // =============================================================================

        /// Edge vertex tables: local vertex indices for each edge
        /// edgeVertices(g, e) returns {v1, v2} - the two vertices of edge e in geometry g
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
                {0, 1}, {1, 2}, {2, 0},  // Base edges
                {0, 3}, {1, 3}, {2, 3}   // Side edges
            }};
            /// Edge vertices for Cube: 12 edges
            inline constexpr std::array<std::pair<int, int>, 12> Cube = {{
                {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
                {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
                {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
            }};
        } // namespace edge_table

        /// Face vertex tables: local vertex indices for each face
        namespace face_table {
            /// Face vertices for Tetrahedron: 4 triangular faces
            /// Face i is opposite to vertex i
            inline constexpr std::array<std::array<int, 3>, 4> Tetrahedron = {{
                {{1, 2, 3}},  // Face opposite vertex 0
                {{0, 3, 2}},  // Face opposite vertex 1
                {{0, 1, 3}},  // Face opposite vertex 2
                {{0, 2, 1}}   // Face opposite vertex 3
            }};
            /// Face vertices for Cube: 6 quadrilateral faces
            /// Ordering: -z, +z, -y, +y, -x, +x (matches MFEM convention)
            inline constexpr std::array<std::array<int, 4>, 6> Cube = {{
                {{0, 1, 2, 3}},  // Bottom (-z)
                {{4, 7, 6, 5}},  // Top (+z) - note: reversed for proper orientation
                {{0, 4, 7, 3}},  // Front (-y)
                {{1, 2, 6, 5}},  // Back (+y)
                {{0, 3, 7, 4}},  // Left (-x)
                {{1, 5, 6, 2}}   // Right (+x)
            }};
        } // namespace face_table

        /// Get local vertex indices for an edge
        /// @param g Geometry type
        /// @param edgeIdx Edge index (0 to numEdges(g)-1)
        /// @return Pair of local vertex indices {v1, v2}, or {0, 0} if invalid
        inline std::pair<int, int> edgeVertices(Geometry g, int edgeIdx)
        {
            const std::pair<int, int>* table = nullptr;
            int nEdges = 0;

            switch (g)
            {
            case Geometry::Triangle:
                table = edge_table::Triangle.data();
                nEdges = 3;
                break;
            case Geometry::Square:
                table = edge_table::Square.data();
                nEdges = 4;
                break;
            case Geometry::Tetrahedron:
                table = edge_table::Tetrahedron.data();
                nEdges = 6;
                break;
            case Geometry::Cube:
                table = edge_table::Cube.data();
                nEdges = 12;
                break;
            default:
                return {0, 0};
            }

            if (edgeIdx < 0 || edgeIdx >= nEdges)
            {
                return {0, 0};
            }

            return table[edgeIdx];
        }

        /// Get local vertex indices for a face
        /// @param g Geometry type
        /// @param faceIdx Face index (0 to numFaces(g)-1)
        /// @return Array of local vertex indices, empty if invalid
        inline std::vector<int> faceVertices(Geometry g, int faceIdx)
        {
            std::vector<int> result;

            switch (g)
            {
            case Geometry::Tetrahedron:
                if (faceIdx >= 0 && faceIdx < 4)
                {
                    result.assign(face_table::Tetrahedron[faceIdx].begin(),
                                  face_table::Tetrahedron[faceIdx].end());
                }
                break;
            case Geometry::Cube:
                if (faceIdx >= 0 && faceIdx < 6)
                {
                    result.assign(face_table::Cube[faceIdx].begin(),
                                  face_table::Cube[faceIdx].end());
                }
                break;
            case Geometry::Triangle:
            case Geometry::Square:
                // 2D elements: the face is the element itself
                for (int i = 0; i < numVertices(g); ++i)
                {
                    result.push_back(i);
                }
                break;
            default:
                break;
            }

            return result;
        }

    } // namespace geom

} // namespace mpfem

#endif // MPFEM_GEOMETRY_HPP
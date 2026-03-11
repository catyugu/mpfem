#ifndef MPFEM_GEOMETRY_HPP
#define MPFEM_GEOMETRY_HPP

#include <cstdint>
#include <string_view>
#include <array>

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

    } // namespace geom

} // namespace mpfem

#endif // MPFEM_GEOMETRY_HPP
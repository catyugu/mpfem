#ifndef MPFEM_GEOMETRY_HPP
#define MPFEM_GEOMETRY_HPP

#include "core/types.hpp"
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>


namespace mpfem {

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
    enum class Geometry : std::uint8_t {
        Point = 0, ///< 0D: Point
        Segment = 1, ///< 1D: Line segment (2 vertices)
        Triangle = 2, ///< 2D: Triangle (3 vertices)
        Square = 3, ///< 2D: Quadrilateral (4 vertices)
        Tetrahedron = 4, ///< 3D: Tetrahedron (4 vertices)
        Cube = 5, ///< 3D: Hexahedron (8 vertices)

        // Sentinel
        Invalid = 255
    };

    // =============================================================================
    // Geometry traits
    // =============================================================================

    namespace geom {

        /// Get the dimension of a geometry type
        constexpr int dim(Geometry g)
        {
            switch (g) {
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
            switch (g) {
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
            switch (g) {
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

            switch (g) {

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

            switch (g) {

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

            switch (g) {

            case Geometry::Cube:
                return 1; // Center node

            default:
                return 0;
            }
        }

        /// Get the number of 2D faces for a geometry type (absolute dimension)
        constexpr int numFaces(Geometry g)
        {
            switch (g) {
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

        /// Get the number of facets for a geometry type (relative dimension = dim-1)
        constexpr int numFacets(Geometry g)
        {
            switch (g) {
            case Geometry::Point:
                return 0;
            case Geometry::Segment:
                return 2;
            case Geometry::Triangle:
                return 3;
            case Geometry::Square:
                return 4;
            case Geometry::Tetrahedron:
                return 4;
            case Geometry::Cube:
                return 6;
            default:
                return 0;
            }
        }

        /// Get the geometry type of 2D faces (absolute dimension)
        constexpr Geometry faceGeometry(Geometry g, int /*faceIdx*/)
        {
            switch (g) {
            case Geometry::Triangle:
            case Geometry::Square:
                return g;
            case Geometry::Tetrahedron:
                return Geometry::Triangle;
            case Geometry::Cube:
                return Geometry::Square;
            default:
                return Geometry::Invalid;
            }
        }

        /// Get the geometry type of facets (relative dimension = dim-1)
        constexpr Geometry facetGeometry(Geometry g, int /*facetIdx*/)
        {
            switch (g) {
            case Geometry::Segment:
                return Geometry::Point;
            case Geometry::Triangle:
            case Geometry::Square:
                return Geometry::Segment;
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
            switch (g) {
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
        inline constexpr std::array<std::array<Real, 1>, 2> refCoords_Segment = {{{{-1.0}}, {{1.0}}}};

        /// Reference vertex coordinates for Triangle: (0,0), (1,0), (0,1)
        inline constexpr std::array<std::array<Real, 2>, 3> refCoords_Triangle = {{{{0.0, 0.0}}, {{1.0, 0.0}}, {{0.0, 1.0}}}};

        /// Reference vertex coordinates for Square: [-1,1] x [-1,1]
        inline constexpr std::array<std::array<Real, 2>, 4> refCoords_Square = {{{{-1.0, -1.0}}, {{1.0, -1.0}}, {{1.0, 1.0}}, {{-1.0, 1.0}}}};

        /// Reference vertex coordinates for Tetrahedron
        inline constexpr std::array<std::array<Real, 3>, 4> refCoords_Tetrahedron = {{{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};

        /// Reference vertex coordinates for Cube: [-1,1]^3
        inline constexpr std::array<std::array<Real, 3>, 8> refCoords_Cube = {{{{-1.0, -1.0, -1.0}}, {{1.0, -1.0, -1.0}}, {{1.0, 1.0, -1.0}}, {{-1.0, 1.0, -1.0}}, {{-1.0, -1.0, 1.0}}, {{1.0, -1.0, 1.0}}, {{1.0, 1.0, 1.0}}, {{-1.0, 1.0, 1.0}}}};

        // =============================================================================
        // Reference element topology tables
        // =============================================================================

        /// Edge vertex tables: local vertex indices for each edge
        /// edgeVertices(g, e) returns {v1, v2} - the two vertices of edge e in geometry g
        namespace edge_table {
            /// Edge vertices for Triangle: 3 edges
            inline constexpr std::array<std::pair<int, int>, 3> Triangle = {{{0, 1}, {2, 0}, {1, 2}}};
            /// Edge vertices for Square: 4 edges
            inline constexpr std::array<std::pair<int, int>, 4> Square = {{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
            /// Edge vertices for Tetrahedron: 6 edges
            inline constexpr std::array<std::pair<int, int>, 6> Tetrahedron = {{
                {0, 1}, {0, 2}, {1, 2}, // Base edges
                {0, 3}, {1, 3}, {2, 3} // Side edges
            }};
            /// Edge vertices for Cube: 12 edges
            inline constexpr std::array<std::pair<int, int>, 12> Cube = {{
                {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
                {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
                {0, 4}, {1, 5}, {2, 6}, {3, 7} // Vertical edges
            }};
        } // namespace edge_table

        /// Face vertex tables: local vertex indices for each face
        namespace face_table {
            /// Face vertices for Tetrahedron: 4 triangular faces
            /// Face i is opposite to vertex i
            inline constexpr std::array<std::array<int, 3>, 4> Tetrahedron = {{
                {{1, 2, 3}}, // Face opposite vertex 0
                {{0, 3, 2}}, // Face opposite vertex 1
                {{0, 1, 3}}, // Face opposite vertex 2
                {{0, 2, 1}} // Face opposite vertex 3
            }};
            /// Face vertices for Cube: 6 quadrilateral faces
            /// Ordering: -z, +z, -y, +y, -x, +x (matches MFEM convention)
            inline constexpr std::array<std::array<int, 4>, 6> Cube = {{
                {{0, 1, 2, 3}}, // Bottom (-z)
                {{4, 7, 6, 5}}, // Top (+z) - note: reversed for proper orientation
                {{0, 4, 7, 3}}, // Front (-y)
                {{1, 2, 6, 5}}, // Back (+y)
                {{0, 3, 7, 4}}, // Left (-x)
                {{1, 5, 6, 2}} // Right (+x)
            }};
        } // namespace face_table

        // =============================================================================
        // Edge midpoint reference coordinates (for second-order elements)
        // =============================================================================

        /// Edge midpoint reference coordinates for Triangle2
        /// Edge ordering matches edge_table::Triangle
        /// Coordinates are (xi, eta) in reference triangle
        inline constexpr std::array<std::array<Real, 2>, 3> edgeMidpoint_Triangle = {{
            {{0.5, 0.0}}, // Edge 0: between vertices 0-1
            {{0.0, 0.5}}, // Edge 1: between vertices 2-0
            {{0.5, 0.5}} // Edge 2: between vertices 1-2
        }};

        /// Edge midpoint reference coordinates for Square2
        /// Edge ordering matches edge_table::Square
        inline constexpr std::array<std::array<Real, 2>, 4> edgeMidpoint_Square = {{
            {{0.0, -1.0}}, // Edge 0: bottom (vertices 0-1)
            {{1.0, 0.0}}, // Edge 1: right (vertices 1-2)
            {{0.0, 1.0}}, // Edge 2: top (vertices 2-3)
            {{-1.0, 0.0}} // Edge 3: left (vertices 3-0)
        }};

        /// Edge midpoint reference coordinates for Tetrahedron2
        /// Edge ordering matches edge_table::Tetrahedron
        inline constexpr std::array<std::array<Real, 3>, 6> edgeMidpoint_Tetrahedron = {{
            {{0.5, 0.0, 0.0}}, // Edge 0: between vertices 0-1
            {{0.0, 0.5, 0.0}}, // Edge 1: between vertices 0-2
            {{0.5, 0.5, 0.0}}, // Edge 2: between vertices 1-2
            {{0.0, 0.0, 0.5}}, // Edge 3: between vertices 0-3
            {{0.5, 0.0, 0.5}}, // Edge 4: between vertices 1-3
            {{0.0, 0.5, 0.5}} // Edge 5: between vertices 2-3
        }};

        /// Edge midpoint reference coordinates for Cube2
        /// Edge ordering matches edge_table::Cube
        inline constexpr std::array<std::array<Real, 3>, 12> edgeMidpoint_Cube = {{
            {{0.0, -1.0, -1.0}}, // Edge 0: bottom front
            {{1.0, 0.0, -1.0}}, // Edge 1: bottom right
            {{0.0, 1.0, -1.0}}, // Edge 2: bottom back
            {{-1.0, 0.0, -1.0}}, // Edge 3: bottom left
            {{0.0, -1.0, 1.0}}, // Edge 4: top front
            {{1.0, 0.0, 1.0}}, // Edge 5: top right
            {{0.0, 1.0, 1.0}}, // Edge 6: top back
            {{-1.0, 0.0, 1.0}}, // Edge 7: top left
            {{-1.0, -1.0, 0.0}}, // Edge 8: front left vertical
            {{1.0, -1.0, 0.0}}, // Edge 9: front right vertical
            {{1.0, 1.0, 0.0}}, // Edge 10: back right vertical
            {{-1.0, 1.0, 0.0}} // Edge 11: back left vertical
        }};

        // =============================================================================
        // Face center reference coordinates (for second-order elements)
        // =============================================================================

        /// Face center reference coordinates for Square2 (single center point)
        /// Center of the reference square [-1,1] x [-1,1]
        inline constexpr std::array<Real, 2> faceCenter_Square = {{0.0, 0.0}};

        /// Face center reference coordinates for Cube2 (6 face centers)
        /// Face ordering matches face_table::Cube: -z, +z, -y, +y, -x, +x
        inline constexpr std::array<std::array<Real, 3>, 6> faceCenter_Cube = {{
            {{0.0, 0.0, -1.0}}, // Face 0: bottom (-z)
            {{0.0, 0.0, 1.0}}, // Face 1: top (+z)
            {{0.0, -1.0, 0.0}}, // Face 2: front (-y)
            {{0.0, 1.0, 0.0}}, // Face 3: back (+y)
            {{-1.0, 0.0, 0.0}}, // Face 4: left (-x)
            {{1.0, 0.0, 0.0}} // Face 5: right (+x)
        }};

        // =============================================================================
        // Volume center reference coordinates (for second-order elements)
        // =============================================================================

        /// Volume center reference coordinates for Cube2 (single center point)
        /// Center of the reference cube [-1,1]^3
        inline constexpr std::array<Real, 3> volumeCenter_Cube = {{0.0, 0.0, 0.0}};

        /// Square2: center of the element (same as face center for 2D)
        inline constexpr std::array<Real, 2> center_Square = {{0.0, 0.0}};

        // =============================================================================
        // Face edge tables: edge indices for each face
        // =============================================================================

        namespace face_edge_table {
            /// Face edges for Tetrahedron: 4 triangular faces
            /// Each face has 3 edges
            /// Edge ordering follows edge_table::Tetrahedron
            inline constexpr std::array<std::array<int, 3>, 4> Tetrahedron = {{
                {{2, 5, 4}}, // Face 0: opposite vertex 0, edges (1-2), (2-3), (1-3)
                {{3, 5, 1}}, // Face 1: opposite vertex 1, edges (0-3), (2-3), (0-2)
                {{0, 4, 3}}, // Face 2: opposite vertex 2, edges (0-1), (1-3), (0-3)
                {{1, 2, 0}} // Face 3: opposite vertex 3, edges (0-2), (1-2), (0-1)
            }};

            /// Face edges for Cube: 6 quadrilateral faces
            /// Each face has 4 edges
            /// Edge ordering follows edge_table::Cube
            inline constexpr std::array<std::array<int, 4>, 6> Cube = {{
                {{0, 1, 2, 3}}, // Face 0: bottom (-z)
                {{4, 7, 6, 5}}, // Face 1: top (+z)
                {{0, 8, 4, 11}}, // Face 2: front (-y)
                {{2, 10, 6, 9}}, // Face 3: back (+y)
                {{3, 11, 7, 10}}, // Face 4: left (-x)
                {{1, 9, 5, 8}} // Face 5: right (+x)
            }};
        } // namespace face_edge_table

        // =============================================================================
        // Helper functions for second-order elements
        // =============================================================================

        /// Get edge midpoint reference coordinates for a geometry type
        /// @param g Geometry type
        /// @param edgeIdx Edge index
        /// @param coords Output coordinates (size = dim)
        /// @return true if successful
        inline bool getEdgeMidpointCoords(Geometry g, int edgeIdx, Real* coords)
        {
            switch (g) {
            case Geometry::Segment:
                if (edgeIdx == 0) {
                    coords[0] = 0.0;
                    return true;
                }
                return false;
            case Geometry::Triangle:
                if (edgeIdx >= 0 && edgeIdx < 3) {
                    coords[0] = edgeMidpoint_Triangle[edgeIdx][0];
                    coords[1] = edgeMidpoint_Triangle[edgeIdx][1];
                    return true;
                }
                return false;
            case Geometry::Square:
                if (edgeIdx >= 0 && edgeIdx < 4) {
                    coords[0] = edgeMidpoint_Square[edgeIdx][0];
                    coords[1] = edgeMidpoint_Square[edgeIdx][1];
                    return true;
                }
                return false;
            case Geometry::Tetrahedron:
                if (edgeIdx >= 0 && edgeIdx < 6) {
                    coords[0] = edgeMidpoint_Tetrahedron[edgeIdx][0];
                    coords[1] = edgeMidpoint_Tetrahedron[edgeIdx][1];
                    coords[2] = edgeMidpoint_Tetrahedron[edgeIdx][2];
                    return true;
                }
                return false;
            case Geometry::Cube:
                if (edgeIdx >= 0 && edgeIdx < 12) {
                    coords[0] = edgeMidpoint_Cube[edgeIdx][0];
                    coords[1] = edgeMidpoint_Cube[edgeIdx][1];
                    coords[2] = edgeMidpoint_Cube[edgeIdx][2];
                    return true;
                }
                return false;
            default:
                return false;
            }
        }

        /// Get edge indices for a 2D face (absolute dimension)
        /// @param g Geometry type
        /// @param faceIdx Face index
        /// @return Vector of edge indices for the face
        inline std::vector<int> faceEdges(Geometry g, int faceIdx)
        {
            std::vector<int> result;

            switch (g) {
            case Geometry::Tetrahedron:
                if (faceIdx >= 0 && faceIdx < 4) {
                    result.assign(face_edge_table::Tetrahedron[faceIdx].begin(),
                        face_edge_table::Tetrahedron[faceIdx].end());
                }
                break;
            case Geometry::Cube:
                if (faceIdx >= 0 && faceIdx < 6) {
                    result.assign(face_edge_table::Cube[faceIdx].begin(),
                        face_edge_table::Cube[faceIdx].end());
                }
                break;
            case Geometry::Triangle:
                if (faceIdx == 0) {
                    result = {0, 1, 2};
                }
                break;
            case Geometry::Square:
                if (faceIdx == 0) {
                    result = {0, 1, 2, 3};
                }
                break;
            default:
                break;
            }

            return result;
        }

        /// Get edge indices for a facet (relative dimension = dim-1)
        /// @param g Geometry type
        /// @param facetIdx Facet index
        /// @return Vector of edge indices for the facet
        inline std::vector<int> facetEdges(Geometry g, int facetIdx)
        {
            if (g == Geometry::Tetrahedron || g == Geometry::Cube) {
                return faceEdges(g, facetIdx);
            }

            if (g == Geometry::Triangle || g == Geometry::Square) {
                if (facetIdx >= 0 && facetIdx < numEdges(g)) {
                    return {facetIdx};
                }
                return {};
            }

            return {};
        }

        /// Get all node reference coordinates for a geometry type with given order
        /// @param g Geometry type
        /// @param order Element order (1 = linear, 2 = quadratic)
        /// @return Vector of reference coordinates for all nodes
        ///
        /// Node ordering for second-order elements:
        /// - Triangle2: 3 corners + 3 edge midpoints = 6 nodes (no face center for simplex)
        /// - Square2:   4 corners + 4 edge midpoints + 1 center = 9 nodes
        /// - Tetrahedron2: 4 corners + 6 edge midpoints = 10 nodes (no face/volume center for simplex)
        /// - Cube2:     8 corners + 12 edge midpoints + 6 face centers + 1 volume center = 27 nodes
        inline std::vector<std::vector<Real>> nodeCoords(Geometry g, int order)
        {
            std::vector<std::vector<Real>> coords;
            int d = dim(g);

            // Add corner vertex coordinates
            switch (g) {
            case Geometry::Segment:
                coords.push_back({-1.0});
                coords.push_back({1.0});
                break;
            case Geometry::Triangle:
                coords.push_back({0.0, 0.0});
                coords.push_back({1.0, 0.0});
                coords.push_back({0.0, 1.0});
                break;
            case Geometry::Square:
                coords.push_back({-1.0, -1.0});
                coords.push_back({1.0, -1.0});
                coords.push_back({1.0, 1.0});
                coords.push_back({-1.0, 1.0});
                break;
            case Geometry::Tetrahedron:
                coords.push_back({0.0, 0.0, 0.0});
                coords.push_back({1.0, 0.0, 0.0});
                coords.push_back({0.0, 1.0, 0.0});
                coords.push_back({0.0, 0.0, 1.0});
                break;
            case Geometry::Cube:
                coords.push_back({-1.0, -1.0, -1.0});
                coords.push_back({1.0, -1.0, -1.0});
                coords.push_back({1.0, 1.0, -1.0});
                coords.push_back({-1.0, 1.0, -1.0});
                coords.push_back({-1.0, -1.0, 1.0});
                coords.push_back({1.0, -1.0, 1.0});
                coords.push_back({1.0, 1.0, 1.0});
                coords.push_back({-1.0, 1.0, 1.0});
                break;
            default:
                return coords;
            }

            // Add edge midpoints for order >= 2
            if (order >= 2) {
                int nEdges = numEdges(g);
                for (int e = 0; e < nEdges; ++e) {
                    std::vector<Real> mp(d);
                    getEdgeMidpointCoords(g, e, mp.data());
                    coords.push_back(std::move(mp));
                }

                // Add face center for Square2 (tensor product element)
                if (g == Geometry::Square) {
                    coords.push_back({center_Square[0], center_Square[1]});
                }

                // Add face centers and volume center for Cube2 (tensor product element)
                if (g == Geometry::Cube) {
                    // Add 6 face centers
                    for (int f = 0; f < 6; ++f) {
                        coords.push_back({faceCenter_Cube[f][0],
                            faceCenter_Cube[f][1],
                            faceCenter_Cube[f][2]});
                    }
                    // Add volume center
                    coords.push_back({volumeCenter_Cube[0],
                        volumeCenter_Cube[1],
                        volumeCenter_Cube[2]});
                }
            }

            // TODO: For order >= 3, add additional interior nodes

            return coords;
        }

        /// Get face center reference coordinates for Cube2
        /// @param faceIdx Face index (0 to 5)
        /// @param coords Output coordinates (size = 3)
        /// @return true if successful
        inline bool getFaceCenterCoords(Geometry g, int faceIdx, Real* coords)
        {
            if (g == Geometry::Cube && faceIdx >= 0 && faceIdx < 6) {
                coords[0] = faceCenter_Cube[faceIdx][0];
                coords[1] = faceCenter_Cube[faceIdx][1];
                coords[2] = faceCenter_Cube[faceIdx][2];
                return true;
            }
            return false;
        }

        /// Get volume center reference coordinates for Cube2
        /// @param coords Output coordinates (size = 3)
        inline void getVolumeCenterCoords(Geometry g, Real* coords)
        {
            if (g == Geometry::Cube) {
                coords[0] = volumeCenter_Cube[0];
                coords[1] = volumeCenter_Cube[1];
                coords[2] = volumeCenter_Cube[2];
            }
        }

        /// Get local vertex indices for an edge
        /// @param g Geometry type
        /// @param edgeIdx Edge index (0 to numEdges(g)-1)
        /// @return Pair of local vertex indices {v1, v2}, or {0, 0} if invalid
        inline std::pair<int, int> edgeVertices(Geometry g, int edgeIdx)
        {
            const std::pair<int, int>* table = nullptr;
            int nEdges = 0;

            switch (g) {
            case Geometry::Segment:
                if (edgeIdx == 0) {
                    return {0, 1};
                }
                return {0, 0};
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

            if (edgeIdx < 0 || edgeIdx >= nEdges) {
                return {0, 0};
            }

            return table[edgeIdx];
        }

        /// Get local vertex indices for a 2D face (absolute dimension)
        /// @param g Geometry type
        /// @param faceIdx Face index (0 to numFaces(g)-1)
        /// @return Array of local vertex indices, empty if invalid
        inline std::vector<int> faceVertices(Geometry g, int faceIdx)
        {
            std::vector<int> result;

            switch (g) {
            case Geometry::Tetrahedron:
                if (faceIdx >= 0 && faceIdx < 4) {
                    result.assign(face_table::Tetrahedron[faceIdx].begin(),
                        face_table::Tetrahedron[faceIdx].end());
                }
                break;
            case Geometry::Cube:
                if (faceIdx >= 0 && faceIdx < 6) {
                    result.assign(face_table::Cube[faceIdx].begin(),
                        face_table::Cube[faceIdx].end());
                }
                break;
            case Geometry::Triangle:
                if (faceIdx == 0) {
                    result = {0, 1, 2};
                }
                break;
            case Geometry::Square:
                if (faceIdx == 0) {
                    result = {0, 1, 2, 3};
                }
                break;
            default:
                break;
            }

            return result;
        }

        /// Get local vertex indices for a facet (relative dimension = dim-1)
        /// @param g Geometry type
        /// @param facetIdx Facet index
        /// @return Array of local vertex indices, empty if invalid
        inline std::vector<int> facetVertices(Geometry g, int facetIdx)
        {
            if (g == Geometry::Tetrahedron || g == Geometry::Cube) {
                return faceVertices(g, facetIdx);
            }

            if (g == Geometry::Triangle || g == Geometry::Square) {
                if (facetIdx >= 0 && facetIdx < numEdges(g)) {
                    auto [v0, v1] = edgeVertices(g, facetIdx);
                    return {v0, v1};
                }
                return {};
            }

            if (g == Geometry::Segment) {
                if (facetIdx == 0) {
                    return {0};
                }
                if (facetIdx == 1) {
                    return {1};
                }
                return {};
            }

            return {};
        }

        struct FaceToVolumeAffineMap {
            Matrix3 A = Matrix3::Zero();
            Vector3 b = Vector3::Zero();
        };

        inline bool getFaceToVolumeAffineMap(Geometry volumeGeom, int localFaceIdx, FaceToVolumeAffineMap& out)
        {
            out = FaceToVolumeAffineMap {};

            if (volumeGeom == Geometry::Tetrahedron) {
                switch (localFaceIdx) {
                case 0:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        -1.0, -1.0, 0.0;
                    out.b << 0.0, 0.0, 1.0;
                    return true;
                case 1:
                    out.A << 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        -1.0, -1.0, 0.0;
                    out.b << 0.0, 0.0, 1.0;
                    return true;
                case 2:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        -1.0, -1.0, 0.0;
                    out.b << 0.0, 0.0, 1.0;
                    return true;
                case 3:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 0.0, 0.0;
                    return true;
                default:
                    return false;
                }
            }

            if (volumeGeom == Geometry::Cube) {
                switch (localFaceIdx) {
                case 0:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 0.0, -1.0;
                    return true;
                case 1:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 0.0, 1.0;
                    return true;
                case 2:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0;
                    out.b << 0.0, -1.0, 0.0;
                    return true;
                case 3:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0;
                    out.b << 0.0, 1.0, 0.0;
                    return true;
                case 4:
                    out.A << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0;
                    out.b << -1.0, 0.0, 0.0;
                    return true;
                case 5:
                    out.A << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0;
                    out.b << 1.0, 0.0, 0.0;
                    return true;
                default:
                    return false;
                }
            }

            if (volumeGeom == Geometry::Triangle) {
                switch (localFaceIdx) {
                case 0:
                    out.A << 1.0, 0.0, 0.0,
                        -1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 1.0, 0.0;
                    return true;
                case 1:
                    out.A << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 0.0, 0.0;
                    return true;
                case 2:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 0.0, 0.0;
                    return true;
                default:
                    return false;
                }
            }

            if (volumeGeom == Geometry::Square) {
                switch (localFaceIdx) {
                case 0:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, -1.0, 0.0;
                    return true;
                case 1:
                    out.A << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 1.0, 0.0, 0.0;
                    return true;
                case 2:
                    out.A << 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << 0.0, 1.0, 0.0;
                    return true;
                case 3:
                    out.A << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
                    out.b << -1.0, 0.0, 0.0;
                    return true;
                default:
                    return false;
                }
            }

            return false;
        }

    } // namespace geom

} // namespace mpfem

#endif // MPFEM_GEOMETRY_HPP
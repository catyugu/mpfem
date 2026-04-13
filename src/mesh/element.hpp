#ifndef MPFEM_ELEMENT_HPP
#define MPFEM_ELEMENT_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"
#include <span>
#include <vector>

namespace mpfem {

    /**
     * @brief Element struct representing a mesh element (topology only, non-owning).
     */
    struct Element {
        Geometry geometry = Geometry::Invalid;
        std::span<const Index> vertices;
        Index attribute = 0;
        int order = 1;

        /// Get spatial dimension of the element
        int dim() const { return geom::dim(geometry); }

        /// Get number of vertices
        int numVertices() const { return static_cast<int>(vertices.size()); }

        /// Get number of edges (from geometry)
        int numEdges() const { return geom::numEdges(geometry); }

        /// Get number of faces (from geometry)
        int numFaces() const { return geom::numFaces(geometry); }

        /// Check if element is a volume element
        bool isVolume() const { return geom::isVolume(geometry); }

        /// Check if element is a surface element
        bool isSurface() const { return geom::isSurface(geometry); }

        /// Get vertex index
        Index vertex(int i) const { return vertices[i]; }

        /// Get number of corner vertices (first-order nodes)
        int numCorners() const { return geom::numCorners(geometry); }

        /// Get global vertex indices for an edge
        std::pair<Index, Index> edgeVertices(int edgeIdx) const
        {
            auto local = geom::edgeVertices(geometry, edgeIdx);
            return {vertices[local.first], vertices[local.second]};
        }

        /// Get global vertex indices for a face
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

        /// Get the geometry type of a face
        Geometry faceGeometry(int faceIdx) const
        {
            return geom::faceGeometry(geometry, faceIdx);
        }
    };

} // namespace mpfem

#endif // MPFEM_ELEMENT_HPP

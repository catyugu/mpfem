#ifndef MPFEM_MPHTXT_READER_HPP
#define MPFEM_MPHTXT_READER_HPP

#include "mesh/mesh.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace mpfem {

/**
 * @brief Reader for COMSOL mphtxt mesh format.
 * 
 * Parses mphtxt files and creates a Mesh object.
 * Supports:
 * - 2D and 3D meshes
 * - Element types: tri/quad (2D boundary), tet/hex (3D volume)
 * - Domain and boundary attributes
 * - First and second order elements (tri2, tet2, edg2, etc.)
 * 
 * Note: Prism and Pyramid elements are NOT supported and will throw an exception.
 * 
 * Second-order element node ordering (COMSOL convention):
 * - Segment2:  V0 V1 E01                          (2 corners + 1 edge midpoint)
 * - Triangle2: V0 V1 V2 E01 E12 E20               (3 corners + 3 edge midpoints)
 * - Tetrahedron2: V0 V1 V2 V3 E01 E02 E12 E03 E13 E23 (4 corners + 6 edge midpoints)
 * 
 * Node reordering for Tetrahedron2:
 * COMSOL edge midpoint order: E01, E02, E12, E03, E13, E23 (nodes 4-9)
 * mpfem edge order:           E01, E12, E20, E03, E13, E23
 * Note: E02 and E20 are the same edge (vertices 0-2)
 * Reordering: swap nodes 5 and 6, then reorder 7,8,9 to 7,8,9 (no change needed for last three)
 */
class MphtxtReader {
public:
    /// Element block info from mphtxt file
    struct ElementBlock {
        std::string typeName;           ///< Element type name (vtx, edg, edg2, tri, tri2, tet, tet2, etc.)
        int numVertsPerElem = 0;        ///< Number of vertices per element
        int order = 1;                  ///< Element order (1 = linear, 2 = quadratic)
        std::vector<std::vector<Index>> elements;  ///< Element connectivity
        std::vector<Index> geomIndices; ///< Geometric entity indices (domain/boundary IDs)
    };

    /// Parsed data from mphtxt file
    struct ParsedData {
        int sdim = 3;                               ///< Spatial dimension
        std::vector<std::array<Real, 3>> vertices;  ///< Vertex coordinates
        std::vector<ElementBlock> blocks;           ///< Element blocks
    };

    /**
     * @brief Read a mesh from an mphtxt file.
     * @param filename Path to the mphtxt file
     * @return Mesh object
     * @throws MeshException if unsupported element types (Prism, Pyramid) are found
     */
    static Mesh read(const std::string& filename) {
        MphtxtReader reader;
        return reader.readFile(filename);
    }

    /**
     * @brief Parse mphtxt file without creating mesh (for inspection).
     * @param filename Path to the mphtxt file
     * @return Parsed data structure
     */
    static ParsedData parse(const std::string& filename) {
        MphtxtReader reader;
        return reader.parseFile(filename);
    }

private:
    /// Read and create mesh
    Mesh readFile(const std::string& filename) {
        LOG_INFO << "Reading mesh from " << filename;
        
        auto data = parseFile(filename);
        
        // Create mesh
        Mesh mesh;
        mesh.setDim(data.sdim);
        mesh.reserveVertices(static_cast<Index>(data.vertices.size()));
        
        // Add vertices
        for (const auto& v : data.vertices) {
            mesh.addVertex(Vertex(v[0], v[1], v[2], data.sdim));
        }
        
        // Process element blocks
        Index numVolumeElems = 0;
        Index numBdrElems = 0;
        
        for (const auto& block : data.blocks) {
            Geometry geom = getGeometryType(block.typeName, block.numVertsPerElem, data.sdim);
            
            // Skip vertex and edge elements - they are not volume or boundary elements
            if (geom == Geometry::Point || geom == Geometry::Segment) {
                LOG_DEBUG << "Skipping " << block.elements.size() << " " << block.typeName << " elements";
                continue;
            }
            
            bool isBoundary = isBoundaryElement(geom, data.sdim);
            int order = block.order;
            
            // Node reordering for second-order elements: COMSOL to mpfem
            // 
            // COMSOL edge ordering (from edge_table):
            // - Triangle: edges are (1,2), (2,0), (0,1) -> E12, E20, E01
            // - Tetrahedron: edges are (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
            //                -> E01, E12, E20, E03, E13, E23
            //
            // mpfem shape function expects:
            // - Triangle: V0 V1 V2 E01 E12 E20 (edge midpoints for edges 0-1, 1-2, 2-0)
            // - Tetrahedron: V0 V1 V2 V3 E01 E12 E20 E03 E13 E23
            //
            // Triangle2 reordering: {V0,V1,V2, E12,E20,E01} -> {V0,V1,V2, E01,E12,E20}
            //                       Mapping: swap nodes 3 and 5
            // Tetrahedron2 reordering: {V0,V1,V2,V3, E01,E12,E20,E03,E13,E23} 
            //                          -> {V0,V1,V2,V3, E01,E12,E20,E03,E13,E23}
            //                          Note: E02 and E20 are same edge, swap nodes 5 and 6
            static const Index tri2Reorder[] = {0, 1, 2, 3, 5, 4};
            static const Index tet2Reorder[] = {0, 1, 2, 3, 4, 6, 5, 7, 8, 9};
            bool needReorderTri = (geom == Geometry::Triangle && order == 2);
            bool needReorderTet = (geom == Geometry::Tetrahedron && order == 2);
            
            if (isBoundary) {
                mesh.reserveBdrElements(mesh.numBdrElements() + 
                    static_cast<Index>(block.elements.size()));
                for (size_t i = 0; i < block.elements.size(); ++i) {
                    // COMSOL uses 0-based indices in mphtxt, convert to 1-based
                    Index attr = 0;
                    if (i < block.geomIndices.size()) {
                        attr = block.geomIndices[i] + 1;
                    }
                    const auto& elemConn = block.elements[i];
                    
                    // Apply reordering for second-order elements
                    if (needReorderTri && elemConn.size() == 6) {
                        std::vector<Index> reordered(6);
                        for (int j = 0; j < 6; ++j) {
                            reordered[j] = elemConn[tri2Reorder[j]];
                        }
                        mesh.addBdrElement(geom, reordered, attr, order);
                    } else if (needReorderTet && elemConn.size() == 10) {
                        std::vector<Index> reordered(10);
                        for (int j = 0; j < 10; ++j) {
                            reordered[j] = elemConn[tet2Reorder[j]];
                        }
                        mesh.addBdrElement(geom, reordered, attr, order);
                    } else {
                        mesh.addBdrElement(geom, elemConn, attr, order);
                    }
                    numBdrElems++;
                }
            } else {
                mesh.reserveElements(mesh.numElements() + 
                    static_cast<Index>(block.elements.size()));
                for (size_t i = 0; i < block.elements.size(); ++i) {
                    // COMSOL domain indices in mphtxt are already 1-based, no conversion needed
                    Index attr = 0;
                    if (i < block.geomIndices.size()) {
                        attr = block.geomIndices[i];
                    }
                    const auto& elemConn = block.elements[i];
                    
                    // Apply reordering for second-order elements
                    if (needReorderTri && elemConn.size() == 6) {
                        std::vector<Index> reordered(6);
                        for (int j = 0; j < 6; ++j) {
                            reordered[j] = elemConn[tri2Reorder[j]];
                        }
                        mesh.addElement(geom, reordered, attr, order);
                    } else if (needReorderTet && elemConn.size() == 10) {
                        std::vector<Index> reordered(10);
                        for (int j = 0; j < 10; ++j) {
                            reordered[j] = elemConn[tet2Reorder[j]];
                        }
                        mesh.addElement(geom, reordered, attr, order);
                    } else {
                        mesh.addElement(geom, elemConn, attr, order);
                    }
                    numVolumeElems++;
                }
            }
        }
        
        LOG_INFO << "Mesh loaded: " << mesh.numVertices() << " vertices, "
                 << numVolumeElems << " volume elements, "
                    << numBdrElems << " boundary elements";
        
        mesh.buildTopology();

        return mesh;
    }

    /// Parse mphtxt file
    ParsedData parseFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw FileException("Cannot open file: " + filename);
        }
        
        ParsedData data;
        std::string line;
        
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            
            // Look for spatial dimension - exact match with "# sdim"
            if (trimmed.find("# sdim") != std::string::npos) {
                std::istringstream iss(trimmed);
                iss >> data.sdim;
                LOG_DEBUG << "Parsed sdim = " << data.sdim;
            }
            
            // Look for vertex count - must be "mesh vertices" not "elements"
            if (trimmed.find("# number of mesh vertices") != std::string::npos) {
                Index numVertices = 0;
                std::istringstream iss(trimmed);
                iss >> numVertices;
                LOG_DEBUG << "Expecting " << numVertices << " vertices";
                
                // Skip to coordinate section
                while (std::getline(file, line)) {
                    if (line.find("# Mesh vertex coordinates") != std::string::npos) {
                        break;
                    }
                }
                
                // Read coordinates
                data.vertices.reserve(numVertices);
                Index count = 0;
                while (count < numVertices && std::getline(file, line)) {
                    std::string coordLine = trim(line);
                    if (coordLine.empty() || coordLine[0] == '#') continue;
                    
                    std::istringstream iss(coordLine);
                    std::array<Real, 3> coords{0.0, 0.0, 0.0};
                    
                    if (data.sdim == 3) {
                        iss >> coords[0] >> coords[1] >> coords[2];
                    } else if (data.sdim == 2) {
                        iss >> coords[0] >> coords[1];
                    } else {
                        iss >> coords[0];
                    }
                    
                    data.vertices.push_back(coords);
                    count++;
                }
                LOG_DEBUG << "Read " << count << " vertices";
            }
            
            // Look for element type blocks
            if (trimmed.find("# Type #") != std::string::npos) {
                ElementBlock block = parseElementBlock(file, trimmed);
                if (!block.elements.empty()) {
                    data.blocks.push_back(std::move(block));
                }
            }
        }
        
        file.close();
        return data;
    }

    /// Parse an element block
    ElementBlock parseElementBlock(std::ifstream& file, const std::string& /*headerLine*/) {
        ElementBlock block;
        std::string line;
        
        // Skip empty lines after "# Type #N" header
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (!trimmed.empty()) break;
        }
        
        // Parse type name: format is "<len> <typename> # type name"
        // e.g., "3 vtx # type name" means type name is "vtx" with length 3
        line = trim(line);
        std::istringstream iss(line);
        std::string token;
        iss >> token;  // Skip count (e.g., "3" in "3 vtx")
        if (iss >> token) {
            block.typeName = token;
            // Detect order from type name
            block.order = detectOrder(block.typeName);
        }
        LOG_DEBUG << "Parsing element block: type=" << block.typeName << ", order=" << block.order;
        
        // Read number of vertices per element
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            
            if (trimmed.find("# number of vertices per element") != std::string::npos ||
                trimmed.find("vertices per element") != std::string::npos) {
                std::istringstream tiss(trimmed);
                tiss >> block.numVertsPerElem;
            } else {
                // Try to parse as number directly
                std::istringstream tiss(trimmed);
                tiss >> block.numVertsPerElem;
            }
            break;
        }
        LOG_DEBUG << "Vertices per element: " << block.numVertsPerElem;
        
        // Read number of elements
        Index numElements = 0;
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            
            if (trimmed.find("# number of elements") != std::string::npos) {
                std::istringstream tiss(trimmed);
                tiss >> numElements;
            } else {
                std::istringstream tiss(trimmed);
                tiss >> numElements;
            }
            
            // Skip to elements section
            while (std::getline(file, line)) {
                if (line.find("# Elements") != std::string::npos) {
                    break;
                }
            }
            
            // Read elements
            block.elements.reserve(numElements);
            Index count = 0;
            while (count < numElements && std::getline(file, line)) {
                std::string trimmed = trim(line);
                if (trimmed.empty() || trimmed[0] == '#') continue;
                
                std::vector<Index> elemVertices;
                std::istringstream viss(trimmed);
                Index v;
                while (viss >> v && elemVertices.size() < static_cast<size_t>(block.numVertsPerElem)) {
                    elemVertices.push_back(v);
                }
                
                // Handle case where vertices span multiple lines
                while (elemVertices.size() < static_cast<size_t>(block.numVertsPerElem) && 
                       std::getline(file, line)) {
                    trimmed = trim(line);
                    if (trimmed.empty() || trimmed[0] == '#') continue;
                    
                    std::istringstream viss2(trimmed);
                    while (viss2 >> v && elemVertices.size() < static_cast<size_t>(block.numVertsPerElem)) {
                        elemVertices.push_back(v);
                    }
                }
                
                if (elemVertices.size() == static_cast<size_t>(block.numVertsPerElem)) {
                    block.elements.push_back(std::move(elemVertices));
                    count++;
                }
            }
            LOG_DEBUG << "Read " << block.elements.size() << " elements of type " << block.typeName;
            
            // Look for geometric entity indices
            while (std::getline(file, line)) {
                std::string trimmed = trim(line);
                if (trimmed.find("# number of geometric entity indices") != std::string::npos) {
                    Index numIndices = 0;
                    std::istringstream tiss(trimmed);
                    tiss >> numIndices;
                    LOG_DEBUG << "Found " << numIndices << " geometric entity indices";
                    
                    // Skip the comment line "# Geometric entity indices"
                    std::getline(file, line);
                    
                    // Read indices - one per line
                    block.geomIndices.reserve(numIndices);
                    for (Index i = 0; i < numIndices && std::getline(file, line); ++i) {
                        std::istringstream iiss(trim(line));
                        Index idx;
                        if (iiss >> idx) {
                            block.geomIndices.push_back(idx);
                        }
                    }
                    LOG_DEBUG << "Read " << block.geomIndices.size() << " indices";
                    
                    break;
                } else if (trimmed.find("# Type #") != std::string::npos) {
                    LOG_DEBUG << "Found next Type block while looking for geometric indices";
                    break;
                } else if (trimmed.find("# ---------") != std::string::npos) {
                    // Start of new object
                    break;
                }
            }
            
            break;
        }
        
        return block;
    }

    /// Detect element order from type name
    int detectOrder(const std::string& typeName) {
        std::string lower = toLower(typeName);
        // Check for second-order indicator
        if (lower.find("2") != std::string::npos && 
            (lower.find("tri2") != std::string::npos ||
             lower.find("tet2") != std::string::npos ||
             lower.find("edg2") != std::string::npos ||
             lower.find("quad2") != std::string::npos ||
             lower.find("hex2") != std::string::npos)) {
            return 2;
        }
        return 1;
    }

    /// Get geometry type from type name and vertex count
    /// @throws MeshException for unsupported element types (Prism, Pyramid)
    Geometry getGeometryType(const std::string& typeName, int numVerts, int sdim) {
        std::string lower = toLower(typeName);
        
        // Check for unsupported types first
        if (lower.find("prism") != std::string::npos || lower.find("wedge") != std::string::npos) {
            throw MeshException("Prism/Wedge elements are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        if (lower.find("pyr") != std::string::npos) {
            throw MeshException("Pyramid elements are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        
        // Check for prism by vertex count (6 vertices in 3D for linear prism, 
        // but we also need to avoid confusing with Triangle2 which has 6 nodes)
        // Prism linear: 6 vertices, Prism quadratic: 15 vertices
        // Triangle2: 6 vertices but sdim would be 2 for boundary or 3 for boundary in 3D mesh
        if (numVerts == 6 && sdim == 3 && lower.find("tri") == std::string::npos) {
            throw MeshException("Prism elements (6 vertices in 3D) are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        if (numVerts == 5 && sdim == 3) {
            throw MeshException("Pyramid elements (5 vertices) are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        
        // Check by type name first (more reliable)
        if (lower.find("vtx") != std::string::npos) {
            return Geometry::Point;
        }
        if (lower.find("edg") != std::string::npos || lower.find("lin") != std::string::npos) {
            return Geometry::Segment;
        }
        if (lower.find("tri") != std::string::npos) {
            return Geometry::Triangle;
        }
        if (lower.find("quad") != std::string::npos) {
            return Geometry::Square;
        }
        if (lower.find("tet") != std::string::npos) {
            return Geometry::Tetrahedron;
        }
        if (lower.find("hex") != std::string::npos) {
            return Geometry::Cube;
        }
        
        return Geometry::Invalid;
    }

    /// Check if geometry is a boundary element (codim-1)
    bool isBoundaryElement(Geometry geom, int sdim) {
        if (sdim == 3) {
            return geom == Geometry::Triangle || geom == Geometry::Square;
        } else if (sdim == 2) {
            return geom == Geometry::Segment;
        } else if (sdim == 1) {
            return geom == Geometry::Point;
        }
        return false;
    }

    /// Trim whitespace from string
    static std::string trim(const std::string& str) {
        size_t start = 0;
        while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
            start++;
        }
        size_t end = str.size();
        while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
            end--;
        }
        return str.substr(start, end - start);
    }

    /// Convert string to lowercase
    static std::string toLower(const std::string& str) {
        std::string result = str;
        for (char& c : result) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }
};

}  // namespace mpfem

#endif  // MPFEM_MPHTXT_READER_HPP

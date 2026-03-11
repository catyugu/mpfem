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
 * - First and second order elements
 * 
 * Note: Prism and Pyramid elements are NOT supported and will throw an exception.
 */
class MphtxtReader {
public:
    /// Element block info from mphtxt file
    struct ElementBlock {
        std::string typeName;           ///< Element type name (vtx, edg, tri, tet, etc.)
        int numVertsPerElem = 0;        ///< Number of vertices per element
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
        LOG_INFO("Reading mesh from " << filename);
        
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
            bool isBoundary = isBoundaryElement(geom, data.sdim);
            
            if (isBoundary) {
                mesh.reserveBdrElements(mesh.numBdrElements() + 
                    static_cast<Index>(block.elements.size()));
                for (size_t i = 0; i < block.elements.size(); ++i) {
                    Index attr = 0;
                    if (i < block.geomIndices.size()) {
                        attr = block.geomIndices[i];
                    }
                    mesh.addBdrElement(geom, block.elements[i], attr);
                    numBdrElems++;
                }
            } else {
                mesh.reserveElements(mesh.numElements() + 
                    static_cast<Index>(block.elements.size()));
                for (size_t i = 0; i < block.elements.size(); ++i) {
                    Index attr = 0;
                    if (i < block.geomIndices.size()) {
                        attr = block.geomIndices[i];
                    }
                    mesh.addElement(geom, block.elements[i], attr);
                    numVolumeElems++;
                }
            }
        }
        
        LOG_INFO("Mesh loaded: " << mesh.numVertices() << " vertices, "
                 << numVolumeElems << " volume elements, "
                 << numBdrElems << " boundary elements");
        
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
            // Look for spatial dimension
            if (line.find("# sdim") != std::string::npos) {
                std::istringstream iss(line);
                iss >> data.sdim;
            }
            
            // Look for vertex coordinates section
            if (line.find("# number of mesh vertices") != std::string::npos) {
                Index numVertices = 0;
                std::istringstream iss(line);
                iss >> numVertices;
                
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
                    std::string trimmed = trim(line);
                    if (trimmed.empty() || trimmed[0] == '#') continue;
                    
                    std::istringstream iss(trimmed);
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
            }
            
            // Look for element type blocks
            if (line.find("# Type") != std::string::npos) {
                ElementBlock block = parseElementBlock(file, line);
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
        
        // Parse type name from header
        // Format: "# Type <n>" followed by type name
        std::getline(file, line);
        line = trim(line);
        
        // Extract type name (second token)
        std::istringstream iss(line);
        std::string token;
        iss >> token;  // Skip count
        if (iss >> token) {
            block.typeName = token;
        }
        
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
        
        // Read number of elements
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            
            Index numElements = 0;
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
            
            // Look for geometric entity indices
            while (std::getline(file, line)) {
                std::string trimmed = trim(line);
                if (trimmed.find("# number of geometric entity indices") != std::string::npos) {
                    Index numIndices = 0;
                    std::istringstream tiss(trimmed);
                    tiss >> numIndices;
                    
                    // Skip empty lines
                    while (std::getline(file, line)) {
                        if (!trim(line).empty()) break;
                    }
                    
                    // Read indices
                    block.geomIndices.reserve(numIndices);
                    Index idxCount = 0;
                    do {
                        std::istringstream iiss(trim(line));
                        Index idx;
                        while (iiss >> idx && idxCount < numIndices) {
                            block.geomIndices.push_back(idx);
                            idxCount++;
                        }
                    } while (idxCount < numIndices && std::getline(file, line));
                    
                    break;
                } else if (trimmed.find("# Type") != std::string::npos) {
                    // Start of next block, push back line indicator
                    // and break (this will be handled by outer loop)
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
        if (numVerts == 5) {
            throw MeshException("Pyramid elements (5 vertices) are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        if (numVerts == 6 && sdim == 3) {
            throw MeshException("Prism elements (6 vertices) are not supported. "
                               "Only tri/quad/tet/hex elements are supported.");
        }
        
        // Supported types
        if (lower.find("vtx") != std::string::npos || numVerts == 1) {
            return Geometry::Point;
        }
        if (lower.find("edg") != std::string::npos || lower.find("lin") != std::string::npos || numVerts == 2) {
            return Geometry::Segment;
        }
        if (lower.find("tri") != std::string::npos || numVerts == 3) {
            return Geometry::Triangle;
        }
        if (lower.find("quad") != std::string::npos || (numVerts == 4 && sdim == 2)) {
            return Geometry::Square;
        }
        if (lower.find("tet") != std::string::npos || (numVerts == 4 && sdim == 3)) {
            return Geometry::Tetrahedron;
        }
        if (lower.find("hex") != std::string::npos || numVerts == 8) {
            return Geometry::Cube;
        }
        
        // Fallback based on vertex count and dimension
        if (numVerts == 4 && sdim == 3) return Geometry::Tetrahedron;
        if (numVerts == 4 && sdim == 2) return Geometry::Square;
        if (numVerts == 8) return Geometry::Cube;
        
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
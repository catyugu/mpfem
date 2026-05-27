#include "io/mphtxt_reader.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/string_utils.hpp"

#include <cctype>
#include <sstream>

namespace mpfem {

    Mesh MphtxtReader::read(const std::string& filename, Real scaleFactor)
    {
        MphtxtReader reader;
        return reader.readFile(filename, scaleFactor);
    }

    MphtxtReader::ParsedData MphtxtReader::parse(const std::string& filename)
    {
        MphtxtReader reader;
        return reader.parseFile(filename);
    }

    Mesh MphtxtReader::readFile(const std::string& filename, Real scaleFactor)
    {
        LOG_INFO << "Reading mesh from " << filename;

        auto data = parseFile(filename);

        Mesh mesh;
        mesh.setDim(data.sdim);
        mesh.reserveNodes(static_cast<Index>(data.vertices.size()));

        for (const auto& v : data.vertices) {
            mesh.addNode(v[0] * scaleFactor, v[1] * scaleFactor, v[2] * scaleFactor);
        }

        Index numVolumeElems = 0;
        Index numBdrElems = 0;

        for (const auto& block : data.blocks) {
            Geometry geom = block.geometry;

            // Determine entity dimension from geometry type
            int entityDim = geom::dim(geom);

            // All entities (0D points, 1D edges, 2D faces, 3D volumes) are added by actual dimension
            for (size_t i = 0; i < block.elements.size(); ++i) {
                Index attr = 0;
                if (i < block.geomIndices.size()) {
                    attr = block.geomIndices[i];
                }

                mesh.addEntity(entityDim, geom, block.elements[i], attr, block.order);
                if (entityDim == data.sdim) {
                    numVolumeElems++;
                } else if (entityDim == data.sdim - 1) {
                    numBdrElems++;
                }
            }
        }

        LOG_INFO << "Mesh loaded: " << mesh.numNodes() << " nodes, "
                 << numVolumeElems << " volume elements, "
                 << numBdrElems << " boundary elements";

        mesh.buildTopology();
        return mesh;
    }

    MphtxtReader::ParsedData MphtxtReader::parseFile(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw FileException("Cannot open file: " + filename);
        }

        ParsedData data;
        std::string line;

        while (std::getline(file, line)) {
            const std::string trimmed = strings::trim(line);

            if (trimmed.find("# sdim") != std::string::npos) {
                std::istringstream iss(trimmed);
                iss >> data.sdim;
                LOG_DEBUG << "Parsed sdim = " << data.sdim;
                continue;
            }

            if (trimmed.find("# number of mesh vertices") != std::string::npos) {
                Index numVertices = 0;
                std::istringstream iss(trimmed);
                iss >> numVertices;
                LOG_DEBUG << "Expecting " << numVertices << " vertices";

                while (std::getline(file, line)) {
                    if (line.find("# Mesh vertex coordinates") != std::string::npos) {
                        break;
                    }
                }

                data.vertices.reserve(numVertices);
                Index count = 0;
                while (count < numVertices && std::getline(file, line)) {
                    const std::string coordLine = strings::trim(line);
                    if (coordLine.empty() || coordLine[0] == '#') {
                        continue;
                    }

                    std::istringstream ciss(coordLine);
                    std::array<Real, 3> coords {0.0, 0.0, 0.0};

                    if (data.sdim == 3) {
                        ciss >> coords[0] >> coords[1] >> coords[2];
                    }
                    else if (data.sdim == 2) {
                        ciss >> coords[0] >> coords[1];
                    }
                    else {
                        ciss >> coords[0];
                    }

                    data.vertices.push_back(coords);
                    count++;
                }
                LOG_DEBUG << "Read " << count << " vertices";
                continue;
            }

            if (trimmed.find("# Type #") != std::string::npos) {
                ElementBlock block = parseElementBlock(file, trimmed, data.sdim);
                if (!block.elements.empty()) {
                    data.blocks.push_back(std::move(block));
                }
            }
        }

        return data;
    }

    MphtxtReader::ElementBlock MphtxtReader::parseElementBlock(std::ifstream& file, const std::string& /*headerLine*/, int sdim)
    {
        ElementBlock block;
        std::string line;

        while (std::getline(file, line)) {
            const std::string trimmed = strings::trim(line);
            if (!trimmed.empty()) {
                break;
            }
        }

        line = strings::trim(line);
        std::istringstream typeIss(line);
        std::string token;
        typeIss >> token;
        if (typeIss >> token) {
            block.typeName = token;
            const std::string lower = toLower(block.typeName);
            block.order = detectOrder(lower);
            block.geometry = getGeometryType(lower, block.numVertsPerElem, sdim);
        }
        LOG_DEBUG << "Parsing element block: type=" << block.typeName << ", order=" << block.order;

        while (std::getline(file, line)) {
            const std::string trimmed = strings::trim(line);
            if (trimmed.empty()) {
                continue;
            }

            std::istringstream nvIss(trimmed);
            nvIss >> block.numVertsPerElem;
            break;
        }
        LOG_DEBUG << "Vertices per element: " << block.numVertsPerElem;

        Index numElements = 0;
        while (std::getline(file, line)) {
            const std::string trimmed = strings::trim(line);
            if (trimmed.empty()) {
                continue;
            }

            std::istringstream neIss(trimmed);
            neIss >> numElements;

            while (std::getline(file, line)) {
                if (line.find("# Elements") != std::string::npos) {
                    break;
                }
            }

            block.elements.reserve(numElements);
            Index count = 0;
            while (count < numElements && std::getline(file, line)) {
                std::string trimmedElem = strings::trim(line);
                if (trimmedElem.empty() || trimmedElem[0] == '#') {
                    continue;
                }

                std::vector<Index> elemVertices;
                std::istringstream eiss(trimmedElem);
                Index v = 0;
                while (eiss >> v && elemVertices.size() < static_cast<size_t>(block.numVertsPerElem)) {
                    elemVertices.push_back(v);
                }

                while (elemVertices.size() < static_cast<size_t>(block.numVertsPerElem) && std::getline(file, line)) {
                    trimmedElem = strings::trim(line);
                    if (trimmedElem.empty() || trimmedElem[0] == '#') {
                        continue;
                    }
                    std::istringstream eiss2(trimmedElem);
                    while (eiss2 >> v && elemVertices.size() < static_cast<size_t>(block.numVertsPerElem)) {
                        elemVertices.push_back(v);
                    }
                }

                if (elemVertices.size() == static_cast<size_t>(block.numVertsPerElem)) {
                    block.elements.push_back(std::move(elemVertices));
                    count++;
                }
            }
            LOG_DEBUG << "Read " << block.elements.size() << " elements of type " << block.typeName;

            while (std::getline(file, line)) {
                const std::string trimmedTail = strings::trim(line);
                if (trimmedTail.find("# number of geometric entity indices") != std::string::npos) {
                    Index numIndices = 0;
                    std::istringstream niIss(trimmedTail);
                    niIss >> numIndices;

                    std::getline(file, line);

                    block.geomIndices.reserve(numIndices);
                    for (Index i = 0; i < numIndices && std::getline(file, line); ++i) {
                        std::istringstream giIss(strings::trim(line));
                        Index idx = 0;
                        if (giIss >> idx) {
                            block.geomIndices.push_back(idx);
                        }
                    }
                    break;
                }

                if (trimmedTail.find("# Type #") != std::string::npos || trimmedTail.find("# ---------") != std::string::npos) {
                    break;
                }
            }

            break;
        }

        return block;
    }

    int MphtxtReader::detectOrder(const std::string& lower)
    {
        if (lower.find("2") != std::string::npos && (lower.find("tri2") != std::string::npos || lower.find("tet2") != std::string::npos || lower.find("edg2") != std::string::npos || lower.find("quad2") != std::string::npos || lower.find("hex2") != std::string::npos)) {
            return 2;
        }
        return 1;
    }

    Geometry MphtxtReader::getGeometryType(const std::string& lower, int numVerts, int sdim)
    {

        if (lower.find("prism") != std::string::npos || lower.find("wedge") != std::string::npos) {
            throw MeshException("Prism/Wedge elements are not supported. Only tri/quad/tet/hex elements are supported.");
        }
        if (lower.find("pyr") != std::string::npos) {
            throw MeshException("Pyramid elements are not supported. Only tri/quad/tet/hex elements are supported.");
        }
        if (numVerts == 6 && sdim == 3 && lower.find("tri") == std::string::npos) {
            throw MeshException("Prism elements (6 vertices in 3D) are not supported. Only tri/quad/tet/hex elements are supported.");
        }
        if (numVerts == 5 && sdim == 3) {
            throw MeshException("Pyramid elements (5 vertices) are not supported. Only tri/quad/tet/hex elements are supported.");
        }

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

    std::string MphtxtReader::toLower(const std::string& str)
    {
        std::string result = str;
        for (char& c : result) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }

} // namespace mpfem

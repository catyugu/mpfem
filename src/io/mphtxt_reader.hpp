#ifndef MPFEM_MPHTXT_READER_HPP
#define MPFEM_MPHTXT_READER_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"
#include "mesh/mesh.hpp"


#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace mpfem {

    class MphtxtReader {
    public:
        struct ElementBlock {
            std::string typeName;
            int numVertsPerElem = 0;
            int order = 1;
            std::vector<std::vector<Index>> elements;
            std::vector<Index> geomIndices;
        };

        struct ParsedData {
            int sdim = 3;
            std::vector<std::array<Real, 3>> vertices;
            std::vector<ElementBlock> blocks;
        };

        static Mesh read(const std::string& filename);
        static ParsedData parse(const std::string& filename);

    private:
        Mesh readFile(const std::string& filename);
        ParsedData parseFile(const std::string& filename);
        ElementBlock parseElementBlock(std::ifstream& file, const std::string& headerLine);

        int detectOrder(const std::string& typeName);
        Geometry getGeometryType(const std::string& typeName, int numVerts, int sdim);
        bool isBoundaryElement(Geometry geom, int sdim);

        static std::string toLower(const std::string& str);
    };

} // namespace mpfem

#endif // MPFEM_MPHTXT_READER_HPP

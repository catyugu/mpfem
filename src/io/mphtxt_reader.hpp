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
            Geometry geometry = Geometry::Invalid;
            std::vector<std::vector<Index>> elements;
            std::vector<Index> geomIndices;
        };

        struct ParsedData {
            int sdim = 3;
            std::vector<std::array<Real, 3>> vertices;
            std::vector<ElementBlock> blocks;
        };

        static Mesh read(const std::string& filename, Real scaleFactor = 1.0);
        static ParsedData parse(const std::string& filename);

    private:
        Mesh readFile(const std::string& filename, Real scaleFactor);
        ParsedData parseFile(const std::string& filename);
        ElementBlock parseElementBlock(std::ifstream& file, const std::string& headerLine, int sdim);

        int detectOrder(const std::string& lower);
        Geometry getGeometryType(const std::string& lower, int numVerts, int sdim);

        static std::string toLower(const std::string& str);
    };

} // namespace mpfem

#endif // MPFEM_MPHTXT_READER_HPP

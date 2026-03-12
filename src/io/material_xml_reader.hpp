#ifndef MPFEM_MATERIAL_XML_READER_HPP
#define MPFEM_MATERIAL_XML_READER_HPP

#include "model/material_database.hpp"
#include <string>

namespace mpfem {

/**
 * @brief Reads material properties from material XML.
 * 
 * Parses the material.xml file and populates a MaterialDatabase.
 */
class MaterialXmlReader {
public:
    /**
     * @brief Parses material XML from disk.
     * @param filePath Path to the material XML file.
     * @param database Output database to populate.
     * @throws FileException if file cannot be opened or parsed.
     */
    static void readFromFile(const std::string& filePath, MaterialDatabase& database);

private:
    static void parsePropertySets(const void* materialElement, 
                                  std::map<std::string, double>& target);
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_XML_READER_HPP

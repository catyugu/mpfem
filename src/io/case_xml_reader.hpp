#ifndef MPFEM_CASE_XML_READER_HPP
#define MPFEM_CASE_XML_READER_HPP

#include "model/case_definition.hpp"
#include <string>

namespace mpfem {

/**
 * @brief Reads complete case schema from case XML.
 * 
 * Parses the case.xml file and populates a CaseDefinition structure.
 */
class CaseXmlReader {
public:
    /**
     * @brief Parses case XML from disk.
     * @param filePath Path to the case XML file.
     * @param caseDefinition Output structure to populate.
     * @throws FileException if file cannot be opened or parsed.
     */
    static void readFromFile(const std::string& filePath, CaseDefinition& caseDefinition);

private:
    static void parseIds(const std::string& text, std::set<int>& ids);
    static std::string trim(const std::string& str);
};

}  // namespace mpfem

#endif  // MPFEM_CASE_XML_READER_HPP

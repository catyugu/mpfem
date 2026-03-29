#include "io/material_xml_reader.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include <tinyxml2.h>

namespace mpfem {

namespace {

// Remove unit brackets like [S/m], [K], etc. from value string
std::string stripUnits(const std::string& value) {
    std::string result = value;
    size_t bracketPos = result.find('[');
    if (bracketPos != std::string::npos) {
        result = result.substr(0, bracketPos);
    }
    // Trim whitespace
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.front()))) {
        result.erase(result.begin());
    }
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) {
        result.pop_back();
    }
    return result;
}

// Parse matrix format {'a','b',...} - returns nullopt if not matrix format
std::optional<Matrix3> parseMatrixConstant(const std::string& value) {
    std::string trimmed = value;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) {
        trimmed.erase(trimmed.begin());
    }
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) {
        trimmed.pop_back();
    }
    
    if (trimmed.size() < 2 || trimmed[0] != '{') {
        return std::nullopt;
    }
    
    std::vector<double> values;
    std::string token;
    bool inQuote = false;
    
    for (size_t i = 1; i < trimmed.size(); ++i) {
        char c = trimmed[i];
        if (c == '\'') {
            inQuote = !inQuote;
            if (!inQuote && !token.empty()) {
                try {
                    values.push_back(std::stod(stripUnits(token)));
                } catch (...) {
                    token.clear();
                    continue;
                }
                token.clear();
            }
        } else if (inQuote) {
            token.push_back(c);
        }
    }
    
    if (values.size() == 9) {
        Matrix3 m;
        m << values[0], values[3], values[6],
             values[1], values[4], values[7],
             values[2], values[5], values[8];
        return m;
    } else if (values.size() == 1) {
        return Matrix3::Identity() * values[0];
    }
    
    return std::nullopt;
}

// Parse scalar constant from value string
std::optional<double> parseScalarConstant(const std::string& value) {
    std::string stripped = stripUnits(value);
    try {
        return std::stod(stripped);
    } catch (...) {
        return std::nullopt;
    }
}

}  // anonymous namespace

void MaterialXmlReader::readFromFile(const std::string& filePath, MaterialDatabase& database) {
    database.clear();

    tinyxml2::XMLDocument document;
    if (document.LoadFile(filePath.c_str()) != tinyxml2::XML_SUCCESS) {
        throw FileException("Failed to parse material XML file: " + filePath);
    }

    const tinyxml2::XMLElement* archiveElement = document.FirstChildElement("archive");
    if (!archiveElement) {
        throw FileException("Missing <archive> root node in material XML: " + filePath);
    }

    const tinyxml2::XMLElement* modelElement = archiveElement->FirstChildElement("model");
    if (!modelElement) {
        throw FileException("Missing <model> node in material XML: " + filePath);
    }

    for (const tinyxml2::XMLElement* materialElement = modelElement->FirstChildElement("material");
         materialElement != nullptr;
         materialElement = materialElement->NextSiblingElement("material")) {
        
        MaterialPropertyModel material;

        if (const char* tagAttr = materialElement->Attribute("tag")) {
            material.tag = tagAttr;
        }

        if (const tinyxml2::XMLElement* labelElement = materialElement->FirstChildElement("label")) {
            if (const char* labelText = labelElement->Attribute("label")) {
                material.label = labelText;
            }
        }

        // Parse all property groups and top-level sets
        auto parseSetElement = [&material](const tinyxml2::XMLElement* setElement) {
            const char* nameAttr = setElement->Attribute("name");
            const char* valueAttr = setElement->Attribute("value");
            if (!nameAttr || !valueAttr) return;
            
            std::string name = nameAttr;
            std::string value = valueAttr;
            
            // Check if it's a matrix format {'a','b',...}
            auto matConst = parseMatrixConstant(value);
            if (matConst.has_value()) {
                material.matrixProperties[name] = matConst.value();
                return;
            }
            
            // Check if it's an expression (contains operators)
            // IMPORTANT: Strip units first because unit strings like [kg/m^3] contain
            // '^' which would incorrectly trigger isExpression to return true
            std::string strippedValue = stripUnits(value);
            if (isExpression(strippedValue)) {
                material.matrixExpressions[name] = strippedValue;
                return;
            }
            
            // Otherwise it's a scalar constant
            auto scalarConst = parseScalarConstant(value);
            if (scalarConst.has_value()) {
                material.scalarProperties[name] = scalarConst.value();
            }
        };

        for (const tinyxml2::XMLElement* groupElement = materialElement->FirstChildElement("propertyGroup");
             groupElement != nullptr;
             groupElement = groupElement->NextSiblingElement("propertyGroup")) {
            for (const tinyxml2::XMLElement* setElement = groupElement->FirstChildElement("set");
                 setElement != nullptr;
                 setElement = setElement->NextSiblingElement("set")) {
                parseSetElement(setElement);
            }
        }

        for (const tinyxml2::XMLElement* setElement = materialElement->FirstChildElement("set");
             setElement != nullptr;
             setElement = setElement->NextSiblingElement("set")) {
            parseSetElement(setElement);
        }

        database.addMaterial(material);
        LOG_DEBUG << "Loaded material: " << material.tag;
    }

    if (database.size() == 0) {
        throw FileException("No material blocks found in file: " + filePath);
    }

    LOG_INFO << "Loaded " << database.size() << " materials from " << filePath;
}

}  // namespace mpfem

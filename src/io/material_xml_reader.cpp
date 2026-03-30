#include "io/material_xml_reader.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include <tinyxml2.h>

namespace mpfem {

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
            material.setTag(tagAttr);
        }

        if (const tinyxml2::XMLElement* labelElement = materialElement->FirstChildElement("label")) {
            if (const char* labelText = labelElement->Attribute("label")) {
                material.setLabel(labelText);
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
            std::string trimmed = strings::trim(value);
            if (trimmed.size() >= 2 && trimmed[0] == '{') {
                material.setMatrix(name, value);
            } else {
                material.setScalar(name, value);
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
        LOG_DEBUG << "Loaded material: " << material.tag();
    }

    if (database.size() == 0) {
        throw FileException("No material blocks found in file: " + filePath);
    }

    LOG_INFO << "Loaded " << database.size() << " materials from " << filePath;
}

} // namespace mpfem

#include "io/material_xml_reader.hpp"
#include "io/value_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"

#include <tinyxml2.h>

namespace mpfem {

void MaterialXmlReader::readFromFile(const std::string& filePath, MaterialDatabase& database) {
    database.clear();

    tinyxml2::XMLDocument document;
    const tinyxml2::XMLError loadError = document.LoadFile(filePath.c_str());
    if (loadError != tinyxml2::XML_SUCCESS) {
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

        // Tag
        if (const char* tagAttr = materialElement->Attribute("tag")) {
            material.tag = tagAttr;
        }

        // Label
        if (const tinyxml2::XMLElement* labelElement = materialElement->FirstChildElement("label")) {
            if (const char* labelText = labelElement->Attribute("label")) {
                material.label = labelText;
            }
        }

        // Parse all property groups
        for (const tinyxml2::XMLElement* groupElement = materialElement->FirstChildElement("propertyGroup");
             groupElement != nullptr;
             groupElement = groupElement->NextSiblingElement("propertyGroup")) {
            
            for (const tinyxml2::XMLElement* setElement = groupElement->FirstChildElement("set");
                 setElement != nullptr;
                 setElement = setElement->NextSiblingElement("set")) {
                
                const char* nameAttr = setElement->Attribute("name");
                const char* valueAttr = setElement->Attribute("value");
                if (nameAttr && valueAttr) {
                    double value = 0.0;
                    if (ValueParser::parseFirstNumber(valueAttr, value)) {
                        material.properties[nameAttr] = value;
                    }
                }
            }
        }

        // Parse top-level set elements
        for (const tinyxml2::XMLElement* setElement = materialElement->FirstChildElement("set");
             setElement != nullptr;
             setElement = setElement->NextSiblingElement("set")) {
            
            const char* nameAttr = setElement->Attribute("name");
            const char* valueAttr = setElement->Attribute("value");
            if (nameAttr && valueAttr) {
                double value = 0.0;
                if (ValueParser::parseFirstNumber(valueAttr, value)) {
                    material.properties[nameAttr] = value;
                }
            }
        }

        // Extract specific properties (names match material.xml)
        auto getProperty = [&](const std::string& name) -> double {
            auto it = material.properties.find(name);
            return it != material.properties.end() ? it->second : 0.0;
        };

        material.rho0 = getProperty("rho0");
        material.alpha = getProperty("alpha");
        material.tref = getProperty("Tref");
        material.electricConductivity = getProperty("electricconductivity");
        material.thermalConductivity = getProperty("thermalconductivity");
        material.youngModulus = getProperty("E");
        material.poissonRatio = getProperty("nu");
        material.thermalExpansion = getProperty("thermalexpansioncoefficient");
        material.density = getProperty("density");
        material.heatCapacity = getProperty("heatcapacity");

        database.addMaterial(material);
        LOG_DEBUG("Loaded material: " << material.tag);
    }

    if (database.size() == 0) {
        throw FileException("No material blocks found in file: " + filePath);
    }

    LOG_INFO("Loaded " << database.size() << " materials from " << filePath);
}

}  // namespace mpfem

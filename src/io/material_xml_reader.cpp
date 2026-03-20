#include "io/material_xml_reader.hpp"
#include "io/value_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
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
            
            // Try to parse as matrix first
            auto mat = ValueParser::parseMatrix(value);
            if (mat.has_value()) {
                material.matrixProperties[name] = mat.value();
                // Also store scalar for backward compatibility
                const auto& m = mat.value();
                material.properties[name] = (m(0,0) + m(1,1) + m(2,2)) / 3.0;
            } else {
                // Fallback to scalar
                double scalar = 0.0;
                if (ValueParser::parseFirstNumber(value, scalar)) {
                    material.properties[name] = scalar;
                }
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

        // Extract specific properties
        auto getScalar = [&](const std::string& n) -> std::optional<double> {
            auto it = material.properties.find(n);
            return it != material.properties.end() ? std::optional<double>{it->second} : std::nullopt;
        };
        
        auto getMatrix = [&](const std::string& n) -> std::optional<Matrix3> {
            auto it = material.matrixProperties.find(n);
            return it != material.matrixProperties.end() ? std::optional<Matrix3>{it->second} : std::nullopt;
        };

        // Temperature-dependent resistivity
        material.rho0 = getScalar("rho0");
        material.alpha = getScalar("alpha");
        material.tref = getScalar("Tref");
        
        // Conductivities - always as matrix
        material.electricConductivity = getMatrix("electricconductivity");
        material.thermalConductivity = getMatrix("thermalconductivity");
        
        // Mechanical properties
        material.youngModulus = getScalar("E");
        material.poissonRatio = getScalar("nu");
        material.thermalExpansion = getScalar("thermalexpansioncoefficient");
        material.density = getScalar("density");
        material.heatCapacity = getScalar("heatcapacity");

        database.addMaterial(material);
        LOG_DEBUG << "Loaded material: " << material.tag;
    }

    if (database.size() == 0) {
        throw FileException("No material blocks found in file: " + filePath);
    }

    LOG_INFO << "Loaded " << database.size() << " materials from " << filePath;
}

}  // namespace mpfem

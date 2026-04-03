#ifndef MPFEM_OPERATOR_XML_PARSER_HPP
#define MPFEM_OPERATOR_XML_PARSER_HPP

#include "operator/operator_config.hpp"
#include <string>
#include <tinyxml2.h>

namespace mpfem {

    /**
     * @brief Recursive XML parser for operator configurations.
     *
     * Parses XML like:
     * <Operator type="GMRES">
     *     <Parameters><MaxIterations>500</MaxIterations></Parameters>
     *     <Preconditioner>
     *         <Operator type="Jacobi"/>
     *     </Preconditioner>
     * </Operator>
     */
    class OperatorXmlParser {
    public:
        /**
         * @brief Parse an <Operator> element into an OperatorConfig.
         */
        static OperatorConfig parse(const tinyxml2::XMLElement* element)
        {
            OperatorConfig config;

            // Get type attribute
            const char* typeAttr = element->Attribute("type");
            if (typeAttr) {
                config.type = typeAttr;
            }

            // Parse Parameters child
            if (const tinyxml2::XMLElement* paramsElement = element->FirstChildElement("Parameters")) {
                config.params = parseParameters(paramsElement);
            }

            // Parse all child elements that are not Parameters
            for (const tinyxml2::XMLElement* child = element->FirstChildElement();
                child != nullptr;
                child = child->NextSiblingElement()) {
                const std::string childName = child->Name();
                if (childName == "Parameters")
                    continue;
                if (childName == "Operator") {
                    // Anonymous operator becomes the value of the parent element's first non-Parameters child
                    // This handles <Preconditioner><Operator type="Jacobi"/></Preconditioner>
                    config.children[childName] = parse(child);
                }
                else {
                    // Named child element - check if it contains an Operator
                    if (const tinyxml2::XMLElement* opElement = child->FirstChildElement("Operator")) {
                        config.children[childName] = parse(opElement);
                    }
                }
            }

            return config;
        }

    private:
        static ParameterList parseParameters(const tinyxml2::XMLElement* paramsElement)
        {
            ParameterList params;

            // First, check if Parameters has attributes (e.g., <Parameters MaxIterations="1000"/>)
            // In this case, the attributes ARE the parameters
            for (const tinyxml2::XMLAttribute* attr = paramsElement->FirstAttribute();
                attr != nullptr;
                attr = attr->Next()) {
                const std::string name = attr->Name();
                if (!name.empty()) {
                    double value = std::atof(attr->Value());
                    params.set(name, value);
                }
            }

            // Also check for child elements (e.g., <Parameters><MaxIterations>100</MaxIterations></Parameters>)
            for (const tinyxml2::XMLElement* param = paramsElement->FirstChildElement();
                param != nullptr;
                param = param->NextSiblingElement()) {
                const std::string name = param->Name();

                // Check for attribute first (e.g., <MaxIterations value="500"/>)
                // Then check for text content (e.g., <MaxIterations>500</MaxIterations>)
                const char* attrValue = param->Attribute("value");
                const char* text = param->GetText();

                if (attrValue) {
                    double value = std::atof(attrValue);
                    params.set(name, value);
                }
                else if (text && text[0] != '\0') {
                    double value = std::atof(text);
                    params.set(name, value);
                }
            }
            return params;
        }
    };

} // namespace mpfem

#endif
#include "io/case_xml_reader.hpp"
#include "io/value_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"

#include <tinyxml2.h>

#include <sstream>
#include <cstdlib>

namespace mpfem {

std::string CaseXmlReader::trim(const std::string& str) {
    size_t first = 0;
    while (first < str.size() && std::isspace(static_cast<unsigned char>(str[first]))) {
        ++first;
    }
    size_t last = str.size();
    while (last > first && std::isspace(static_cast<unsigned char>(str[last - 1]))) {
        --last;
    }
    return str.substr(first, last - first);
}

void CaseXmlReader::parseIds(const std::string& text, std::set<int>& ids) {
    ids.clear();

    std::stringstream tokenStream(text);
    std::string token;

    while (std::getline(tokenStream, token, ',')) {
        const std::string trimmedToken = trim(token);
        if (trimmedToken.empty()) continue;

        const size_t rangePos = trimmedToken.find('-');
        if (rangePos == std::string::npos) {
            // Single ID
            int value = std::stoi(trimmedToken);
            if (value >= 1) {
                ids.insert(value);
            }
        } else {
            // Range: e.g., "1-7"
            const std::string beginToken = trim(trimmedToken.substr(0, rangePos));
            const std::string endToken = trim(trimmedToken.substr(rangePos + 1));
            int beginValue = std::stoi(beginToken);
            int endValue = std::stoi(endToken);

            for (int value = beginValue; value <= endValue; ++value) {
                if (value >= 1) {
                    ids.insert(value);
                }
            }
        }
    }
}

void CaseXmlReader::readFromFile(const std::string& filePath, CaseDefinition& caseDefinition) {
    caseDefinition = CaseDefinition();

    tinyxml2::XMLDocument document;
    const tinyxml2::XMLError loadError = document.LoadFile(filePath.c_str());
    if (loadError != tinyxml2::XML_SUCCESS) {
        throw FileException("Failed to parse XML file: " + filePath);
    }

    const tinyxml2::XMLElement* caseElement = document.FirstChildElement("case");
    if (!caseElement) {
        throw FileException("Missing <case> root node in: " + filePath);
    }

    // Case name
    if (const char* nameAttr = caseElement->Attribute("name")) {
        caseDefinition.caseName = nameAttr;
    }

    // Study type
    if (const tinyxml2::XMLElement* studyElement = caseElement->FirstChildElement("study")) {
        if (const char* typeAttr = studyElement->Attribute("type")) {
            caseDefinition.studyType = typeAttr;
        }
    }

    // Paths
    if (const tinyxml2::XMLElement* pathsElement = caseElement->FirstChildElement("paths")) {
        if (const char* meshAttr = pathsElement->Attribute("mesh")) {
            caseDefinition.meshPath = meshAttr;
        }
        if (const char* materialsAttr = pathsElement->Attribute("materials")) {
            caseDefinition.materialsPath = materialsAttr;
        }
        if (const char* comsolResultAttr = pathsElement->Attribute("comsol_result")) {
            caseDefinition.comsolResultPath = comsolResultAttr;
        }
    }

    // Variables
    if (const tinyxml2::XMLElement* variablesElement = caseElement->FirstChildElement("variables")) {
        for (const tinyxml2::XMLElement* varElement = variablesElement->FirstChildElement("var");
             varElement != nullptr;
             varElement = varElement->NextSiblingElement("var")) {
            
            VariableEntry entry;
            if (const char* nameAttr = varElement->Attribute("name")) {
                entry.name = nameAttr;
            }
            if (const char* valueAttr = varElement->Attribute("value")) {
                entry.valueText = valueAttr;
            }
            if (const char* siAttr = varElement->Attribute("si")) {
                ValueParser::parseFirstNumber(siAttr, entry.siValue);
            }
            caseDefinition.variables.push_back(entry);
        }
    }

    // Material assignments
    if (const tinyxml2::XMLElement* materialsElement = caseElement->FirstChildElement("materials")) {
        for (const tinyxml2::XMLElement* assignElement = materialsElement->FirstChildElement("assign");
             assignElement != nullptr;
             assignElement = assignElement->NextSiblingElement("assign")) {
            
            MaterialAssignment assignment;
            if (const char* materialAttr = assignElement->Attribute("material")) {
                assignment.materialTag = materialAttr;
            }
            if (const char* domainsAttr = assignElement->Attribute("domains")) {
                parseIds(domainsAttr, assignment.domainIds);
            }
            caseDefinition.materialAssignments.push_back(assignment);
        }
    }

    // Physics definitions
    for (const tinyxml2::XMLElement* physicsElement = caseElement->FirstChildElement("physics");
         physicsElement != nullptr;
         physicsElement = physicsElement->NextSiblingElement("physics")) {
        
        PhysicsDefinition physics;
        
        if (const char* kindAttr = physicsElement->Attribute("kind")) {
            physics.kind = kindAttr;
        }
        if (const char* orderAttr = physicsElement->Attribute("order")) {
            physics.order = std::atoi(orderAttr);
            if (physics.order < 1) physics.order = 1;
        }

        // Solver configuration
        if (const tinyxml2::XMLElement* solverElement = physicsElement->FirstChildElement("solver")) {
            if (const char* typeAttr = solverElement->Attribute("type")) {
                physics.solver.type = solverTypeFromName(typeAttr);
            }
            if (const char* maxIterAttr = solverElement->Attribute("max_iter")) {
                physics.solver.maxIterations = std::atoi(maxIterAttr);
            }
            if (const char* tolAttr = solverElement->Attribute("tolerance")) {
                physics.solver.relativeTolerance = std::atof(tolAttr);
            }
            if (const char* printLevelAttr = solverElement->Attribute("print_level")) {
                physics.solver.printLevel = std::atoi(printLevelAttr);
            }
            if (const char* dropTolAttr = solverElement->Attribute("drop_tol")) {
                physics.solver.dropTolerance = std::atof(dropTolAttr);
            }
            if (const char* fillFactorAttr = solverElement->Attribute("fill_factor")) {
                physics.solver.fillFactor = std::atoi(fillFactorAttr);
            }
        }

        // Boundary conditions
        for (const tinyxml2::XMLElement* boundaryElement = physicsElement->FirstChildElement("boundary");
             boundaryElement != nullptr;
             boundaryElement = boundaryElement->NextSiblingElement("boundary")) {
            
            BoundaryCondition boundary;
            if (const char* kindAttr = boundaryElement->Attribute("kind")) {
                boundary.kind = kindAttr;
            }
            if (const char* idsAttr = boundaryElement->Attribute("ids")) {
                parseIds(idsAttr, boundary.ids);
            }

            // Parameters
            for (const tinyxml2::XMLElement* paramElement = boundaryElement->FirstChildElement("param");
                 paramElement != nullptr;
                 paramElement = paramElement->NextSiblingElement("param")) {
                
                const char* nameAttr = paramElement->Attribute("name");
                const char* valueAttr = paramElement->Attribute("value");
                if (nameAttr && valueAttr) {
                    boundary.params[nameAttr] = valueAttr;
                }
            }

            physics.boundaries.push_back(boundary);
        }

        // Sources
        for (const tinyxml2::XMLElement* sourceElement = physicsElement->FirstChildElement("source");
             sourceElement != nullptr;
             sourceElement = sourceElement->NextSiblingElement("source")) {
            
            SourceDefinition source;
            if (const char* kindAttr = sourceElement->Attribute("kind")) {
                source.kind = kindAttr;
            }
            if (const char* valueAttr = sourceElement->Attribute("value")) {
                source.valueText = valueAttr;
            }
            if (const char* domainsAttr = sourceElement->Attribute("domains")) {
                parseIds(domainsAttr, source.domainIds);
            }
            physics.sources.push_back(source);
        }

        caseDefinition.physicsDefinitions.push_back(physics);
    }

    // Coupled physics definitions
    for (const tinyxml2::XMLElement* coupledElement = caseElement->FirstChildElement("coupledPhysics");
         coupledElement != nullptr;
         coupledElement = coupledElement->NextSiblingElement("coupledPhysics")) {
        
        CoupledPhysicsDefinition coupling;
        if (const char* nameAttr = coupledElement->Attribute("name")) {
            coupling.name = nameAttr;
        }
        if (const char* kindAttr = coupledElement->Attribute("kind")) {
            coupling.kind = kindAttr;
        }
        if (const char* physicsAttr = coupledElement->Attribute("physics")) {
            // Split comma-separated physics kinds
            std::stringstream ss(physicsAttr);
            std::string item;
            while (std::getline(ss, item, ',')) {
                std::string trimmed = trim(item);
                if (!trimmed.empty()) {
                    coupling.physicsKinds.push_back(trimmed);
                }
            }
        }
        if (const char* domainsAttr = coupledElement->Attribute("domains")) {
            parseIds(domainsAttr, coupling.domainIds);
        }
        caseDefinition.coupledPhysicsDefinitions.push_back(coupling);
    }

    // Coupling configuration
    if (const tinyxml2::XMLElement* couplingConfigElement = caseElement->FirstChildElement("coupling")) {
        if (const char* maxIterAttr = couplingConfigElement->Attribute("max_iter")) {
            caseDefinition.couplingConfig.maxIterations = std::atoi(maxIterAttr);
        }
        if (const char* tolAttr = couplingConfigElement->Attribute("tolerance")) {
            caseDefinition.couplingConfig.tolerance = std::atof(tolAttr);
        }
    }

    LOG_INFO << "Loaded case definition: " << caseDefinition.caseName 
             << " with " << caseDefinition.physicsDefinitions.size() << " physics fields";
}

}  // namespace mpfem

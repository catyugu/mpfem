#include "io/case_xml_reader.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "core/logger.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"

#include <tinyxml2.h>

#include <sstream>
#include <cstdlib>
#include <cmath>

namespace mpfem {

using strings::trim;

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

    std::map<std::string, double> variableValues;
    auto evalExpr = [&variableValues](const char* text, double defaultValue) -> double {
        if (!text) {
            return defaultValue;
        }
        return ExpressionParser::instance().evaluate(text, variableValues);
    };

    // Variables are parsed first so all following numeric attributes can use
    // a single expression pipeline with variable substitution.
    if (const tinyxml2::XMLElement* variablesElement = caseElement->FirstChildElement("variables")) {
        for (const tinyxml2::XMLElement* varElement = variablesElement->FirstChildElement("var");
             varElement != nullptr;
             varElement = varElement->NextSiblingElement("var")) {

            const char* nameAttr = varElement->Attribute("name");
            if (!nameAttr) {
                continue;
            }

            VariableEntry entry;
            entry.name = nameAttr;

            const char* valueAttr = varElement->Attribute("value");
            if (valueAttr) {
                entry.valueText = valueAttr;
            }

            const char* siAttr = varElement->Attribute("si");
            const char* evalText = siAttr ? siAttr : valueAttr;
            entry.siValue = evalExpr(evalText, 0.0);

            variableValues[entry.name] = entry.siValue;
            caseDefinition.variables.push_back(entry);
        }
    }

    // Study type
    if (const tinyxml2::XMLElement* studyElement = caseElement->FirstChildElement("study")) {
        if (const char* typeAttr = studyElement->Attribute("type")) {
            caseDefinition.studyType = typeAttr;
        }

        // Parse <time start="0" end="100" step="10" scheme="BDF1"/>
        if (const tinyxml2::XMLElement* timeElement = studyElement->FirstChildElement("time")) {
            caseDefinition.timeConfig.start = evalExpr(timeElement->Attribute("start"), 0.0);
            caseDefinition.timeConfig.end = evalExpr(timeElement->Attribute("end"), 1.0);
            caseDefinition.timeConfig.step = evalExpr(timeElement->Attribute("step"), 0.01);
            if (const char* schemeAttr = timeElement->Attribute("scheme")) {
                caseDefinition.timeConfig.scheme = schemeAttr;
            }
        }

        // Parse <initialConditions><field kind="..." value="..."/></initialConditions>
        if (const tinyxml2::XMLElement* icElement = studyElement->FirstChildElement("initialConditions")) {
            for (auto* fieldElement = icElement->FirstChildElement("field");
                 fieldElement; fieldElement = fieldElement->NextSiblingElement("field")) {
                InitialCondition ic;
                if (const char* kindAttr = fieldElement->Attribute("kind")) {
                    ic.fieldKind = kindAttr;
                }
                const char* valueAttr = fieldElement->Attribute("value");
                if (!valueAttr) {
                    valueAttr = fieldElement->Attribute("displacement");
                }
                ic.value = evalExpr(valueAttr, 0.0);
                caseDefinition.initialConditions.push_back(ic);
            }
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
        
        CaseDefinition::Physics physics;
        
        if (const char* kindAttr = physicsElement->Attribute("kind")) {
            physics.kind = kindAttr;
        }
        if (const char* orderAttr = physicsElement->Attribute("order")) {
            physics.order = static_cast<int>(std::lround(evalExpr(orderAttr, 1.0)));
            if (physics.order < 1) physics.order = 1;
        }
        // Reference temperature for thermal expansion [K]
        if (const tinyxml2::XMLElement* refTempElement = physicsElement->FirstChildElement("referenceTemperature")) {
            physics.referenceTemperature = evalExpr(refTempElement->Attribute("value"), 293.15);
        }

        // Solver configuration
        if (const tinyxml2::XMLElement* solverElement = physicsElement->FirstChildElement("solver")) {
            if (const char* typeAttr = solverElement->Attribute("type")) {
                physics.solver.type = solverTypeFromName(typeAttr);
            }
            if (const char* maxIterAttr = solverElement->Attribute("max_iter")) {
                physics.solver.maxIterations =
                    static_cast<int>(std::lround(evalExpr(maxIterAttr, physics.solver.maxIterations)));
            }
            if (const char* tolAttr = solverElement->Attribute("tolerance")) {
                physics.solver.relativeTolerance = evalExpr(tolAttr, physics.solver.relativeTolerance);
            }
            if (const char* printLevelAttr = solverElement->Attribute("print_level")) {
                physics.solver.printLevel =
                    static_cast<int>(std::lround(evalExpr(printLevelAttr, physics.solver.printLevel)));
            }
            if (const char* dropTolAttr = solverElement->Attribute("drop_tol")) {
                physics.solver.dropTolerance = evalExpr(dropTolAttr, physics.solver.dropTolerance);
            }
            if (const char* fillFactorAttr = solverElement->Attribute("fill_factor")) {
                physics.solver.fillFactor =
                    static_cast<int>(std::lround(evalExpr(fillFactorAttr, physics.solver.fillFactor)));
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

        std::string kind = physics.kind;
        caseDefinition.physics[kind] = std::move(physics);
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
            caseDefinition.couplingConfig.maxIterations =
                static_cast<int>(std::lround(evalExpr(maxIterAttr, caseDefinition.couplingConfig.maxIterations)));
        }
        if (const char* tolAttr = couplingConfigElement->Attribute("tolerance")) {
            caseDefinition.couplingConfig.tolerance =
                evalExpr(tolAttr, caseDefinition.couplingConfig.tolerance);
        }
    }

    caseDefinition.buildVariableMap();

    LOG_INFO << "Loaded case definition: " << caseDefinition.caseName 
             << " with " << caseDefinition.physics.size() << " physics fields";
}

}  // namespace mpfem

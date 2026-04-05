#include "io/case_xml_reader.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/string_utils.hpp"
#include "solver/solver_factory.hpp"

#include <tinyxml2.h>

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace mpfem {

    using strings::trim;

    namespace {

        std::map<std::string, std::string> parseBoundaryParams(const tinyxml2::XMLElement* boundaryElement)
        {
            std::map<std::string, std::string> params;
            for (const tinyxml2::XMLElement* paramElement = boundaryElement->FirstChildElement("param");
                paramElement != nullptr;
                paramElement = paramElement->NextSiblingElement("param")) {
                const char* nameAttr = paramElement->Attribute("name");
                const char* valueAttr = paramElement->Attribute("value");
                if (nameAttr && valueAttr) {
                    params[nameAttr] = valueAttr;
                }
            }
            return params;
        }

        BoundaryCondition buildBoundaryCondition(const std::string& physicsKind,
            const std::string& boundaryKind,
            std::set<int> ids,
            const std::map<std::string, std::string>& params)
        {
            BoundaryCondition bc;
            bc.ids = std::move(ids);
            bc.parameters = params; // Already contains "value", "h", "T_inf" as needed

            if (physicsKind == "electrostatics") {
                if (boundaryKind == "voltage") {
                    bc.type = "Voltage";
                }
                else if (boundaryKind == "electric_insulation") {
                    bc.type = "ElectricInsulation";
                }
            }
            else if (physicsKind == "heat_transfer") {
                if (boundaryKind == "temperature") {
                    bc.type = "Temperature";
                }
                else if (boundaryKind == "convection") {
                    bc.type = "Convection";
                }
                else if (boundaryKind == "thermal_insulation") {
                    bc.type = "ThermalInsulation";
                }
            }
            else if (physicsKind == "solid_mechanics") {
                if (boundaryKind == "fixed_constraint") {
                    bc.type = "Fixed";
                }
                else if (boundaryKind == "free_boundary") {
                    bc.type = "Free";
                }
            }

            if (bc.type.empty()) {
                throw FileException("Unsupported boundary kind '" + boundaryKind + "' for physics '" + physicsKind + "'");
            }

            return bc;
        }

    } // namespace

    void CaseXmlReader::parseIds(const std::string& text, std::set<int>& ids)
    {
        ids.clear();

        std::stringstream tokenStream(text);
        std::string token;

        while (std::getline(tokenStream, token, ',')) {
            const std::string trimmedToken = trim(token);
            if (trimmedToken.empty())
                continue;

            const size_t rangePos = trimmedToken.find('-');
            if (rangePos == std::string::npos) {
                // Single ID
                int value = std::stoi(trimmedToken);
                if (value >= 1) {
                    ids.insert(value);
                }
            }
            else {
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

    void CaseXmlReader::readFromFile(const std::string& filePath, CaseDefinition& caseDefinition)
    {
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

        auto readParameterMap = [&](const tinyxml2::XMLElement* parametersElement) {
            std::map<std::string, Real> parameters;
            if (!parametersElement) {
                return parameters;
            }

            for (const tinyxml2::XMLElement* parameter = parametersElement->FirstChildElement();
                parameter != nullptr;
                parameter = parameter->NextSiblingElement()) {
                const std::string key = parameter->Name();
                std::string value;
                if (const char* valueAttr = parameter->Attribute("value")) {
                    value = trim(valueAttr);
                }
                else if (const char* text = parameter->GetText()) {
                    value = trim(text);
                }

                if (!key.empty() && !value.empty()) {
                    parameters[key] = static_cast<Real>(std::stod(value));
                }
            }

            return parameters;
        };

        std::function<std::unique_ptr<LinearOperatorConfig>(const tinyxml2::XMLElement*)> readOperatorConfigNode;
        readOperatorConfigNode = [&](const tinyxml2::XMLElement* element) -> std::unique_ptr<LinearOperatorConfig> {
            if (!element) {
                return nullptr;
            }

            const char* typeAttr = element->Attribute("type");
            if (!typeAttr) {
                throw FileException(std::string("Missing operator node type on <") + element->Name() + "> in " + filePath);
            }

            auto node = std::make_unique<LinearOperatorConfig>();
            node->type = operatorTypeFromName(typeAttr);
            node->parameters = readParameterMap(element->FirstChildElement("Parameters"));

            for (const tinyxml2::XMLElement* child = element->FirstChildElement();
                child != nullptr;
                child = child->NextSiblingElement()) {
                const std::string childName = child->Name();
                if (childName == "Preconditioner") {
                    node->preconditioner = readOperatorConfigNode(child);
                    continue;
                }
                if (childName == "LocalSolver") {
                    node->localSolver = readOperatorConfigNode(child);
                    continue;
                }
                if (childName == "CoarseSolver") {
                    node->coarseSolver = readOperatorConfigNode(child);
                    continue;
                }
                if (childName == "Smoother") {
                    node->smoother = readOperatorConfigNode(child);
                }
            }

            return node;
        };

        // Variables are parsed first and stored as expression text.
        // Evaluation is deferred to VariableManager at runtime.
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

                // value attribute stores expression text (e.g., "20[mV]", "k * 2 + 1")
                // si attribute is removed - VariableManager handles unit conversion at runtime
                const char* valueAttr = varElement->Attribute("value");
                if (valueAttr) {
                    entry.valueText = valueAttr;
                }

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
                if (const char* startAttr = timeElement->Attribute("start")) {
                    caseDefinition.timeConfig.start = std::stod(startAttr);
                }
                if (const char* endAttr = timeElement->Attribute("end")) {
                    caseDefinition.timeConfig.end = std::stod(endAttr);
                }
                if (const char* stepAttr = timeElement->Attribute("step")) {
                    caseDefinition.timeConfig.step = std::stod(stepAttr);
                }
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
                    if (valueAttr) {
                        ic.value = std::stod(valueAttr);
                    }
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
                physics.order = static_cast<int>(std::lround(std::stod(orderAttr)));
                if (physics.order < 1)
                    physics.order = 1;
            }
            // Reference temperature for thermal expansion [K]
            if (const tinyxml2::XMLElement* refTempElement = physicsElement->FirstChildElement("referenceTemperature")) {
                if (const char* valueAttr = refTempElement->Attribute("value")) {
                    physics.referenceTemperature = std::stod(valueAttr);
                }
            }

            // Solver configuration
            if (const tinyxml2::XMLElement* solverConfigElement = physicsElement->FirstChildElement("SolverConfiguration")) {
                const tinyxml2::XMLElement* linearSolverElement = solverConfigElement->FirstChildElement("LinearSolver");
                if (!linearSolverElement) {
                    throw FileException("Missing <LinearSolver> in <SolverConfiguration> for physics '" + physics.kind + "'");
                }

                // Parse the root operator configuration
                physics.solver = readOperatorConfigNode(linearSolverElement);
            }
            else {
                throw FileException("Missing <SolverConfiguration> for physics '" + physics.kind + "'");
            }

            // Boundary conditions
            for (const tinyxml2::XMLElement* boundaryElement = physicsElement->FirstChildElement("boundary");
                boundaryElement != nullptr;
                boundaryElement = boundaryElement->NextSiblingElement("boundary")) {

                std::string boundaryKind;
                if (const char* kindAttr = boundaryElement->Attribute("kind")) {
                    boundaryKind = kindAttr;
                }
                if (const char* idsAttr = boundaryElement->Attribute("ids")) {
                    std::set<int> ids;
                    parseIds(idsAttr, ids);

                    if (ids.empty()) {
                        throw FileException("Boundary ids cannot be empty for kind '" + boundaryKind + "'");
                    }

                    const auto params = parseBoundaryParams(boundaryElement);
                    physics.boundaries.push_back(
                        buildBoundaryCondition(physics.kind, boundaryKind, std::move(ids), params));
                }
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
                caseDefinition.couplingConfig.maxIterations = static_cast<int>(std::lround(std::stod(maxIterAttr)));
            }
            if (const char* tolAttr = couplingConfigElement->Attribute("tolerance")) {
                caseDefinition.couplingConfig.tolerance = std::stod(tolAttr);
            }
        }

        LOG_INFO << "Loaded case definition: " << caseDefinition.caseName
                 << " with " << caseDefinition.physics.size() << " physics fields";
    }

} // namespace mpfem

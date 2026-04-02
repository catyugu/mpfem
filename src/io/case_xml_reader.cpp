#include "io/case_xml_reader.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/string_utils.hpp"
#include "expr/expression_parser.hpp"
#include "solver/solver_factory.hpp"

#include <tinyxml2.h>

#include <cctype>
#include <cstdlib>
#include <functional>
#include <sstream>

namespace mpfem {

    using strings::trim;

    namespace {

        std::string normalizeToken(std::string_view text)
        {
            std::string normalized;
            normalized.reserve(text.size());
            for (char ch : text) {
                const unsigned char value = static_cast<unsigned char>(ch);
                if (std::isalnum(value)) {
                    normalized.push_back(static_cast<char>(std::tolower(value)));
                }
            }
            return normalized;
        }

        const std::string& requireBoundaryParam(const std::map<std::string, std::string>& params,
            const std::string& key,
            const std::string& physicsKind,
            const std::string& boundaryKind)
        {
            auto it = params.find(key);
            if (it != params.end()) {
                return it->second;
            }
            throw FileException("Missing boundary parameter '" + key + "' for boundary kind '" + boundaryKind + "' in physics '" + physicsKind + "'");
        }

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
            if (physicsKind == "electrostatics") {
                if (boundaryKind == "voltage") {
                    return VoltageBoundaryCondition {std::move(ids),
                        requireBoundaryParam(params, "value", physicsKind, boundaryKind)};
                }
                if (boundaryKind == "electric_insulation") {
                    return ElectricInsulationBoundaryCondition {std::move(ids)};
                }
            }

            if (physicsKind == "heat_transfer") {
                if (boundaryKind == "temperature") {
                    return TemperatureBoundaryCondition {std::move(ids),
                        requireBoundaryParam(params, "value", physicsKind, boundaryKind)};
                }
                if (boundaryKind == "convection") {
                    return ConvectionBoundaryCondition {
                        std::move(ids),
                        requireBoundaryParam(params, "h", physicsKind, boundaryKind),
                        requireBoundaryParam(params, "T_inf", physicsKind, boundaryKind)};
                }
                if (boundaryKind == "thermal_insulation") {
                    return ThermalInsulationBoundaryCondition {std::move(ids)};
                }
            }

            if (physicsKind == "solid_mechanics") {
                if (boundaryKind == "fixed_constraint") {
                    return FixedConstraintBoundaryCondition {std::move(ids)};
                }
                if (boundaryKind == "free_boundary") {
                    return FreeBoundaryCondition {std::move(ids)};
                }
            }

            throw FileException("Unsupported boundary kind '" + boundaryKind + "' for physics '" + physicsKind + "'");
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

        std::map<std::string, double> variableValues;
        auto evalExpr = [&variableValues](const char* text, double defaultValue) -> double {
            if (!text) {
                return defaultValue;
            }
            return ExpressionParser::instance().evaluate(text, variableValues);
        };

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
                    parameters[key] = static_cast<Real>(evalExpr(value.c_str(), 0.0));
                }
            }

            return parameters;
        };

        auto getParameter = [&](const std::map<std::string, Real>& parameters,
                                std::initializer_list<std::string_view> names) -> const Real* {
            for (std::string_view name : names) {
                const std::string expected = normalizeToken(name);
                for (const auto& [key, value] : parameters) {
                    if (normalizeToken(key) == expected) {
                        return &value;
                    }
                }
            }
            return nullptr;
        };

        auto parseParameterReal = [&](const std::map<std::string, Real>& parameters,
                                      std::initializer_list<std::string_view> names,
                                      double defaultValue) -> double {
            const Real* value = getParameter(parameters, names);
            if (!value) {
                return defaultValue;
            }
            return static_cast<double>(*value);
        };

        auto parseParameterInt = [&](const std::map<std::string, Real>& parameters,
                                     std::initializer_list<std::string_view> names,
                                     int defaultValue) -> int {
            const Real* value = getParameter(parameters, names);
            if (!value) {
                return defaultValue;
            }
            return static_cast<int>(std::lround(static_cast<double>(*value)));
        };

        std::function<PreconditionerConfig*(const tinyxml2::XMLElement*)> readPreconditionerNode;
        readPreconditionerNode = [&](const tinyxml2::XMLElement* element) -> PreconditionerConfig* {
            if (!element) {
                return nullptr;
            }

            const char* typeAttr = element->Attribute("type");
            if (!typeAttr) {
                throw FileException(std::string("Missing preconditioner node type on <") + element->Name() + "> in " + filePath);
            }

            auto* node = new PreconditionerConfig();
            node->type = SolverFactory::preconditionerTypeFromName(typeAttr);
            node->parameters = readParameterMap(element->FirstChildElement("Parameters"));

            for (const tinyxml2::XMLElement* child = element->FirstChildElement();
                child != nullptr;
                child = child->NextSiblingElement()) {
                const std::string childName = child->Name();
                if (childName == "LocalSolver") {
                    node->localSolver = readPreconditionerNode(child);
                    continue;
                }
                if (childName == "CoarseSolver") {
                    node->coarseSolver = readPreconditionerNode(child);
                    continue;
                }
                if (childName == "Smoother") {
                    node->smoother = readPreconditionerNode(child);
                }
            }

            return node;
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
                if (physics.order < 1)
                    physics.order = 1;
            }
            // Reference temperature for thermal expansion [K]
            if (const tinyxml2::XMLElement* refTempElement = physicsElement->FirstChildElement("referenceTemperature")) {
                physics.referenceTemperature = evalExpr(refTempElement->Attribute("value"), 293.15);
            }

            // Solver configuration
            if (const tinyxml2::XMLElement* solverConfigElement = physicsElement->FirstChildElement("SolverConfiguration")) {
                const tinyxml2::XMLElement* linearSolverElement = solverConfigElement->FirstChildElement("LinearSolver");
                if (!linearSolverElement) {
                    throw FileException("Missing <LinearSolver> in <SolverConfiguration> for physics '" + physics.kind + "'");
                }

                const char* linearTypeAttr = linearSolverElement->Attribute("type");
                if (!linearTypeAttr) {
                    throw FileException("Missing LinearSolver type for physics '" + physics.kind + "'");
                }
                physics.solver.solver.type = SolverFactory::linearSolverTypeFromName(linearTypeAttr);

                const auto linearParameters = readParameterMap(linearSolverElement->FirstChildElement("Parameters"));
                physics.solver.solver.maxIterations = parseParameterInt(
                    linearParameters,
                    {"MaxIterations"},
                    physics.solver.solver.maxIterations);
                physics.solver.solver.restart = parseParameterInt(
                    linearParameters,
                    {"Restart"},
                    physics.solver.solver.restart);
                physics.solver.solver.tolerance = parseParameterReal(
                    linearParameters,
                    {"Tolerance", "RelativeTolerance"},
                    physics.solver.solver.tolerance);
                physics.solver.solver.printLevel = parseParameterInt(
                    linearParameters,
                    {"PrintLevel"},
                    physics.solver.solver.printLevel);

                if (const tinyxml2::XMLElement* preconditionerElement = linearSolverElement->FirstChildElement("Preconditioner")) {
                    physics.solver.preconditioner = *readPreconditionerNode(preconditionerElement);
                }
                else {
                    physics.solver.preconditioner.type = PreconditionerType::None;
                    physics.solver.preconditioner.parameters.clear();
                    physics.solver.preconditioner.localSolver = nullptr;
                    physics.solver.preconditioner.coarseSolver = nullptr;
                    physics.solver.preconditioner.smoother = nullptr;
                }
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
                caseDefinition.couplingConfig.maxIterations = static_cast<int>(std::lround(evalExpr(maxIterAttr, caseDefinition.couplingConfig.maxIterations)));
            }
            if (const char* tolAttr = couplingConfigElement->Attribute("tolerance")) {
                caseDefinition.couplingConfig.tolerance = evalExpr(tolAttr, caseDefinition.couplingConfig.tolerance);
            }
        }

        caseDefinition.buildVariableMap();

        LOG_INFO << "Loaded case definition: " << caseDefinition.caseName
                 << " with " << caseDefinition.physics.size() << " physics fields";
    }

} // namespace mpfem

#include "mpfem/config/problem_config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "tinyxml2.h"

#include "mpfem/core/types.hpp"

namespace mpfem {

namespace {

std::string Trim(const std::string& value) {
  auto begin = value.begin();
  while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }

  auto end = value.end();
  while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
    --end;
  }

  return std::string(begin, end);
}

StudyKind ParseStudyKind(const std::string& value) {
  if (value == "steady") {
    return StudyKind::kSteady;
  }
  if (value == "transient") {
    return StudyKind::kTransient;
  }
  if (value == "eigen") {
    return StudyKind::kEigen;
  }
  throw std::runtime_error("Unsupported study type: " + value);
}

std::vector<PhysicsKind> ParsePhysicsList(const std::string& text) {
  std::vector<PhysicsKind> physics;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = Trim(token);
    if (!token.empty()) {
      physics.push_back(PhysicsKindFromString(token));
    }
  }
  return physics;
}

bool IsSubset(const std::vector<int>& subset, const std::vector<int>& superset) {
  if (subset.empty()) {
    return true;
  }
  if (superset.empty()) {
    return true;
  }

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < subset.size() && j < superset.size()) {
    if (subset[i] == superset[j]) {
      ++i;
      ++j;
      continue;
    }
    if (subset[i] > superset[j]) {
      ++j;
      continue;
    }
    return false;
  }
  return i == subset.size();
}

ElectrostaticsBoundaryKind ParseElectrostaticsBoundaryKind(const std::string& value) {
  if (value == "voltage" || value == "dirichlet") {
    return ElectrostaticsBoundaryKind::kVoltage;
  }
  if (value == "normal_current_density" || value == "neumann") {
    return ElectrostaticsBoundaryKind::kNormalCurrentDensity;
  }
  if (value == "electric_insulation" || value == "insulation") {
    return ElectrostaticsBoundaryKind::kElectricInsulation;
  }
  throw std::runtime_error("Unsupported electrostatics boundary kind: " + value);
}

ElectrostaticsSourceKind ParseElectrostaticsSourceKind(const std::string& value) {
  if (value == "volume_charge_density") {
    return ElectrostaticsSourceKind::kVolumeChargeDensity;
  }
  if (value == "volume_current_source" || value == "current_source") {
    return ElectrostaticsSourceKind::kVolumeCurrentSource;
  }
  throw std::runtime_error("Unsupported electrostatics source kind: " + value);
}

HeatBoundaryKind ParseHeatBoundaryKind(const std::string& value) {
  if (value == "temperature" || value == "dirichlet") {
    return HeatBoundaryKind::kTemperature;
  }
  if (value == "heat_flux" || value == "neumann") {
    return HeatBoundaryKind::kHeatFlux;
  }
  if (value == "convection") {
    return HeatBoundaryKind::kConvection;
  }
  if (value == "thermal_insulation" || value == "insulation") {
    return HeatBoundaryKind::kThermalInsulation;
  }
  throw std::runtime_error("Unsupported heat boundary kind: " + value);
}

HeatSourceKind ParseHeatSourceKind(const std::string& value) {
  if (value == "volumetric_heat_source" || value == "heat_source") {
    return HeatSourceKind::kVolumetricHeatSource;
  }
  throw std::runtime_error("Unsupported heat source kind: " + value);
}

SolidMechanicsBoundaryKind ParseSolidBoundaryKind(const std::string& value) {
  if (value == "displacement") {
    return SolidMechanicsBoundaryKind::kDisplacement;
  }
  if (value == "traction") {
    return SolidMechanicsBoundaryKind::kTraction;
  }
  if (value == "fixed" || value == "fixed_constraint") {
    return SolidMechanicsBoundaryKind::kFixedConstraint;
  }
  if (value == "free" || value == "free_boundary") {
    return SolidMechanicsBoundaryKind::kFreeBoundary;
  }
  throw std::runtime_error("Unsupported solid mechanics boundary kind: " + value);
}

SolidMechanicsSourceKind ParseSolidSourceKind(const std::string& value) {
  if (value == "body_force") {
    return SolidMechanicsSourceKind::kBodyForce;
  }
  if (value == "initial_strain") {
    return SolidMechanicsSourceKind::kInitialStrain;
  }
  throw std::runtime_error("Unsupported solid mechanics source kind: " + value);
}

}  // namespace

ProblemDefinition ProblemConfigLoader::LoadFromXml(const std::string& case_file_path) const {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(case_file_path.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to open case file: " + case_file_path);
  }

  const tinyxml2::XMLElement* root = doc.FirstChildElement("case");
  if (root == nullptr) {
    throw std::runtime_error("Invalid case XML: missing <case>");
  }

  ProblemDefinition definition;
  if (const char* name = root->Attribute("name")) {
    definition.name = name;
  }

  if (const tinyxml2::XMLElement* paths = root->FirstChildElement("paths")) {
    if (const char* mesh = paths->Attribute("mesh")) {
      definition.mesh_path = ResolveRelativePath(case_file_path, mesh);
    }
    if (const char* material = paths->Attribute("materials")) {
      definition.material_path = ResolveRelativePath(case_file_path, material);
    }
    if (const char* result = paths->Attribute("comsol_result")) {
      definition.reference_result_path = ResolveRelativePath(case_file_path, result);
    }
  }

  if (const tinyxml2::XMLElement* study = root->FirstChildElement("study")) {
    if (const char* type = study->Attribute("type")) {
      definition.study = ParseStudyKind(type);
    }
  }

  if (const tinyxml2::XMLElement* variables = root->FirstChildElement("variables")) {
    for (const tinyxml2::XMLElement* var = variables->FirstChildElement("var");
         var != nullptr;
         var = var->NextSiblingElement("var")) {
      const char* name = var->Attribute("name");
      if (name == nullptr) {
        continue;
      }
      const char* value = var->Attribute("value");
      const char* si = var->Attribute("si");

      if (value != nullptr) {
        definition.variables_expr[name] = value;
      }
      if (si != nullptr) {
        definition.variables_si[name] = std::stod(si);
      }
    }
  }

  if (const tinyxml2::XMLElement* materials = root->FirstChildElement("materials")) {
    for (const tinyxml2::XMLElement* assign = materials->FirstChildElement("assign");
         assign != nullptr;
         assign = assign->NextSiblingElement("assign")) {
      const char* domains = assign->Attribute("domains");
      const char* material = assign->Attribute("material");
      if (domains == nullptr || material == nullptr) {
        continue;
      }

      for (const int domain_id : ParseIdList(domains)) {
        MaterialAssignment mapping;
        mapping.domain_id = domain_id;
        mapping.material_tag = material;
        definition.materials.push_back(mapping);
      }
    }
  }

  std::unordered_map<PhysicsKind, DomainScope> physics_scopes;

  for (const tinyxml2::XMLElement* physics = root->FirstChildElement("physics");
       physics != nullptr;
       physics = physics->NextSiblingElement("physics")) {
    const char* physics_kind = physics->Attribute("kind");
    if (physics_kind == nullptr) {
      continue;
    }

    PhysicsModel model;
    model.kind = PhysicsKindFromString(physics_kind);

    if (const char* domains = physics->Attribute("domains")) {
      model.scope.domain_ids = ParseIdList(domains);
    }

    switch (model.kind) {
      case PhysicsKind::kElectrostatics: {
        ElectrostaticsPhysicsModel data;

        for (const tinyxml2::XMLElement* bc = physics->FirstChildElement("boundary");
             bc != nullptr;
             bc = bc->NextSiblingElement("boundary")) {
          ElectrostaticsBoundaryCondition condition;
          if (const char* kind = bc->Attribute("kind")) {
            condition.kind = ParseElectrostaticsBoundaryKind(kind);
          }
          if (const char* ids = bc->Attribute("ids")) {
            condition.boundary_ids = ParseIdList(ids);
          }
          if (const char* value = bc->Attribute("value")) {
            condition.value_expr = value;
          }
          data.boundary_conditions.push_back(std::move(condition));
        }

        for (const tinyxml2::XMLElement* source = physics->FirstChildElement("source");
             source != nullptr;
             source = source->NextSiblingElement("source")) {
          ElectrostaticsSourceTerm term;
          if (const char* kind = source->Attribute("kind")) {
            term.kind = ParseElectrostaticsSourceKind(kind);
          }
          if (const char* domains = source->Attribute("domains")) {
            term.domain_ids = ParseIdList(domains);
          }
          if (const char* value = source->Attribute("value")) {
            term.value_expr = value;
          }
          data.sources.push_back(std::move(term));
        }

        model.model = std::move(data);
        break;
      }

      case PhysicsKind::kHeatTransfer: {
        HeatTransferPhysicsModel data;

        for (const tinyxml2::XMLElement* bc = physics->FirstChildElement("boundary");
             bc != nullptr;
             bc = bc->NextSiblingElement("boundary")) {
          HeatBoundaryCondition condition;
          if (const char* kind = bc->Attribute("kind")) {
            condition.kind = ParseHeatBoundaryKind(kind);
          }
          if (const char* ids = bc->Attribute("ids")) {
            condition.boundary_ids = ParseIdList(ids);
          }
          if (const char* value = bc->Attribute("value")) {
            condition.value_expr = value;
          }
          if (const char* aux = bc->Attribute("aux")) {
            condition.aux_value_expr = aux;
          }
          data.boundary_conditions.push_back(std::move(condition));
        }

        for (const tinyxml2::XMLElement* source = physics->FirstChildElement("source");
             source != nullptr;
             source = source->NextSiblingElement("source")) {
          HeatSourceTerm term;
          if (const char* kind = source->Attribute("kind")) {
            term.kind = ParseHeatSourceKind(kind);
          }
          if (const char* domains = source->Attribute("domains")) {
            term.domain_ids = ParseIdList(domains);
          }
          if (const char* value = source->Attribute("value")) {
            term.value_expr = value;
          }
          data.sources.push_back(std::move(term));
        }

        model.model = std::move(data);
        break;
      }

      case PhysicsKind::kSolidMechanics: {
        SolidMechanicsPhysicsModel data;

        for (const tinyxml2::XMLElement* bc = physics->FirstChildElement("boundary");
             bc != nullptr;
             bc = bc->NextSiblingElement("boundary")) {
          SolidMechanicsBoundaryCondition condition;
          if (const char* kind = bc->Attribute("kind")) {
            condition.kind = ParseSolidBoundaryKind(kind);
          }
          if (const char* ids = bc->Attribute("ids")) {
            condition.boundary_ids = ParseIdList(ids);
          }
          if (const char* value = bc->Attribute("value")) {
            condition.value_expr = value;
          }
          data.boundary_conditions.push_back(std::move(condition));
        }

        for (const tinyxml2::XMLElement* source = physics->FirstChildElement("source");
             source != nullptr;
             source = source->NextSiblingElement("source")) {
          SolidMechanicsSourceTerm term;
          if (const char* kind = source->Attribute("kind")) {
            term.kind = ParseSolidSourceKind(kind);
          }
          if (const char* domains = source->Attribute("domains")) {
            term.domain_ids = ParseIdList(domains);
          }
          if (const char* value = source->Attribute("value")) {
            term.value_expr = value;
          }
          data.sources.push_back(std::move(term));
        }

        model.model = std::move(data);
        break;
      }
    }

    physics_scopes[model.kind] = model.scope;
    definition.physics.push_back(std::move(model));
  }

  for (const tinyxml2::XMLElement* coupling = root->FirstChildElement("coupledPhysics");
       coupling != nullptr;
       coupling = coupling->NextSiblingElement("coupledPhysics")) {
    CoupledPhysicsModel model;

    if (const char* name = coupling->Attribute("name")) {
      model.name = name;
    }
    if (const char* kind = coupling->Attribute("kind")) {
      model.kind = CoupledPhysicsKindFromString(kind);
    }
    if (const char* physics = coupling->Attribute("physics")) {
      model.sub_physics = ParsePhysicsList(physics);
    }
    if (const char* domains = coupling->Attribute("domains")) {
      model.scope.domain_ids = ParseIdList(domains);
    }

    if (model.sub_physics.size() < 2) {
      throw std::runtime_error("Each coupledPhysics model must contain at least two sub-physics.");
    }

    for (const PhysicsKind kind : model.sub_physics) {
      const auto found = physics_scopes.find(kind);
      if (found == physics_scopes.end()) {
        throw std::runtime_error("coupledPhysics references undefined sub-physics: " + ToString(kind));
      }
      if (!model.scope.AppliesToAllDomains() && !found->second.AppliesToAllDomains() &&
          !IsSubset(model.scope.domain_ids, found->second.domain_ids)) {
        throw std::runtime_error(
            "coupledPhysics domains must be subset of each sub-physics domain scope.");
      }
    }

    definition.coupled_physics.push_back(std::move(model));
  }

  return definition;
}

std::vector<int> ProblemConfigLoader::ParseIdList(const std::string& text) {
  std::vector<int> ids;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = Trim(token);
    if (token.empty()) {
      continue;
    }

    const std::size_t dash = token.find('-');
    if (dash == std::string::npos) {
      ids.push_back(std::stoi(token));
      continue;
    }

    const int begin = std::stoi(Trim(token.substr(0, dash)));
    const int end = std::stoi(Trim(token.substr(dash + 1)));
    if (end < begin) {
      throw std::runtime_error("Invalid range in ids list: " + token);
    }
    for (int value = begin; value <= end; ++value) {
      ids.push_back(value);
    }
  }

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

std::string ProblemConfigLoader::ResolveRelativePath(const std::string& case_file_path,
                                                     const std::string& candidate_path) {
  const std::filesystem::path candidate(candidate_path);
  if (candidate.is_absolute()) {
    return candidate.lexically_normal().string();
  }

  const std::filesystem::path case_path(case_file_path);
  const std::filesystem::path root = case_path.parent_path();
  return (root / candidate).lexically_normal().string();
}

}  // namespace mpfem

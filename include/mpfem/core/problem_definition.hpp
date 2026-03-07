#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "mpfem/core/types.hpp"
#include "mpfem/physics/coupled_physics_model.hpp"
#include "mpfem/physics/physics_model.hpp"

namespace mpfem {

struct MaterialAssignment {
  int domain_id = -1;
  std::string material_tag;
};

struct ProblemDefinition {
  std::string name;
  std::string mesh_path;
  std::string material_path;
  std::string reference_result_path;
  StudyKind study = StudyKind::kSteady;
  std::vector<PhysicsModel> physics;
  std::vector<CoupledPhysicsModel> coupled_physics;
  std::vector<MaterialAssignment> materials;
  std::map<std::string, double> variables_si;
  std::map<std::string, std::string> variables_expr;
};

struct CaseValidationReport {
  int domains = 0;
  int boundaries = 0;
  int materials = 0;
  std::set<std::string> material_labels;
};

}  // namespace mpfem

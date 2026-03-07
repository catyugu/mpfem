#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <variant>

#include "mpfem/config/problem_config_loader.hpp"
#include "mpfem/material/comsol_material_reader.hpp"
#include "mpfem/mesh/comsol_mesh_reader.hpp"

namespace {

int Fail(const std::string& message) {
  std::cerr << "[FAIL] " << message << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path root =
      (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path(".");

  const std::filesystem::path case_path = root / "cases" / "busbar" / "case.xml";

  try {
    mpfem::ProblemConfigLoader config_loader;
    const mpfem::ProblemDefinition problem = config_loader.LoadFromXml(case_path.string());

    mpfem::ComsolMeshReader mesh_reader;
    const mpfem::ComsolMeshSummary summary = mesh_reader.ReadSummary(problem.mesh_path);

    if (summary.vertex_count != 7360) {
      return Fail("Vertex count mismatch.");
    }
    if (summary.domain_count != 7) {
      return Fail("Domain count mismatch. Expected 7.");
    }
    if (summary.boundary_count != 43) {
      return Fail("Boundary count mismatch. Expected 43.");
    }

    mpfem::ComsolMaterialReader material_reader;
    const auto materials = material_reader.Read(problem.material_path);
    if (materials.size() != 2) {
      return Fail("Material count mismatch. Expected 2.");
    }

    std::set<std::string> labels;
    for (const auto& material : materials) {
      labels.insert(material.label);
    }

    if (!labels.count("Copper")) {
      return Fail("Material label Copper not found.");
    }
    if (!labels.count("Titanium beta-21S")) {
      return Fail("Material label Titanium beta-21S not found.");
    }

    if (problem.physics.size() != 3) {
      return Fail("Physics model count mismatch. Expected 3.");
    }
    for (const auto& model : problem.physics) {
      if (!model.scope.AppliesToAllDomains()) {
        return Fail("Physics model should default to all domains when domains is omitted.");
      }
    }
    if (problem.coupled_physics.size() != 2) {
      return Fail("Coupled physics model count mismatch. Expected 2.");
    }
    if (problem.materials.size() != 7) {
      return Fail("Domain-to-material assignment count mismatch. Expected 7.");
    }

    int boundary_groups = 0;
    int source_groups = 0;
    for (const auto& model : problem.physics) {
      std::visit(
          [&](const auto& concrete_model) {
            boundary_groups += static_cast<int>(concrete_model.boundary_conditions.size());
            source_groups += static_cast<int>(concrete_model.sources.size());
          },
          model.model);
    }
    if (boundary_groups < 7) {
      return Fail("Boundary condition groups are fewer than expected.");
    }
    if (source_groups < 1) {
      return Fail("At least one source term should be defined.");
    }

    std::cout << "[PASS] busbar case parser checks" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    return Fail(std::string("Unhandled exception: ") + ex.what());
  }
}

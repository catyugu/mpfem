#include <filesystem>
#include <iostream>
#include <set>
#include <string>

#include "mpfem/config/problem_config_loader.hpp"
#include "mpfem/material/comsol_material_reader.hpp"
#include "mpfem/mesh/comsol_mesh_reader.hpp"

int main(int argc, char** argv) {
  const std::filesystem::path root =
      (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path(".");
  const std::filesystem::path case_path = root / "cases" / "busbar" / "case.xml";

  try {
    mpfem::ProblemConfigLoader config_loader;
    const mpfem::ProblemDefinition problem = config_loader.LoadFromXml(case_path.string());

    mpfem::ComsolMeshReader mesh_reader;
    const mpfem::ComsolMeshSummary summary = mesh_reader.ReadSummary(problem.mesh_path);

    mpfem::ComsolMaterialReader material_reader;
    const auto materials = material_reader.Read(problem.material_path);

    std::set<std::string> labels;
    for (const auto& material : materials) {
      labels.insert(material.label);
    }

    std::cout << "Busbar mesh summary\n";
    std::cout << "  vertices:   " << summary.vertex_count << "\n";
    std::cout << "  domains:    " << summary.domain_count << "\n";
    std::cout << "  boundaries: " << summary.boundary_count << "\n";
    std::cout << "  materials:  " << materials.size() << "\n";
    std::cout << "  physics:    " << problem.physics.size() << "\n";

    if (summary.domain_count != 7 || summary.boundary_count != 43) {
      std::cerr << "Error: busbar mesh topology does not match expected (7 domains, 43 boundaries).\n";
      return 2;
    }

    if (!labels.count("Copper") || !labels.count("Titanium beta-21S")) {
      std::cerr << "Error: required materials are missing from material.xml.\n";
      return 3;
    }

    if (problem.physics.size() != 3 || problem.coupled_physics.size() != 2) {
      std::cerr << "Error: case.xml physics and coupling definitions are incomplete.\n";
      return 4;
    }

    std::cout << "Validation passed.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Fatal error: " << ex.what() << "\n";
    return 1;
  }
}

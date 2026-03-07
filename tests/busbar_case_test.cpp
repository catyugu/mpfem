#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <variant>

#include "mpfem/config/problem_config_loader.hpp"
#include "mpfem/core/logger.hpp"
#include "mpfem/material/comsol_material_reader.hpp"
#include "mpfem/mesh/comsol_mesh_reader.hpp"

namespace {

int Fail(const std::string& message) {
  std::cerr << "[FAIL] " << message << "\n";
  return 1;
}

int TestMeshSummary(const mpfem::ComsolMeshSummary& summary) {
  std::cout << "Testing mesh summary..." << std::endl;

  if (summary.vertex_count != 7360) {
    return Fail("Vertex count mismatch.");
  }
  if (summary.domain_count != 7) {
    return Fail("Domain count mismatch. Expected 7.");
  }
  if (summary.boundary_count != 43) {
    return Fail("Boundary count mismatch. Expected 43.");
  }

  std::cout << "[PASS] Mesh summary test" << std::endl;
  return 0;
}

int TestFullMeshRead(const mpfem::Mesh& mesh) {
  std::cout << "Testing full mesh read..." << std::endl;

  // 验证节点数
  if (mesh.NodeCount() != 7360) {
    return Fail("Node count mismatch. Expected 7360, got " + std::to_string(mesh.NodeCount()));
  }

  // 验证第一个节点的坐标（从网格文件可知）
  const auto& nodes = mesh.Nodes();
  constexpr double tol = 1e-12;
  if (std::abs(nodes.x[0] - 0.0975) > tol || std::abs(nodes.y[0] - 0.0) > tol ||
      std::abs(nodes.z[0] - 0.1) > tol) {
    return Fail("First node coordinates mismatch.");
  }

  // 验证域单元组
  if (mesh.DomainElements().empty()) {
    return Fail("No domain elements found.");
  }

  // 统计域单元类型
  std::map<mpfem::GeometryType, int> domain_elem_counts;
  for (const auto& group : mesh.DomainElements()) {
    domain_elem_counts[group.type] += group.Count();
    std::cout << "  Domain element type " << static_cast<int>(group.type) << ": " << group.Count()
              << " elements" << std::endl;
  }

  // 验证边界单元组
  if (mesh.BoundaryElements().empty()) {
    return Fail("No boundary elements found.");
  }

  std::map<mpfem::GeometryType, int> boundary_elem_counts;
  for (const auto& group : mesh.BoundaryElements()) {
    boundary_elem_counts[group.type] += group.Count();
    std::cout << "  Boundary element type " << static_cast<int>(group.type) << ": " << group.Count()
              << " elements" << std::endl;
  }

  // 验证域ID
  const auto domain_ids = mesh.GetDomainIds();
  std::cout << "  Domain IDs: ";
  for (int id : domain_ids) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  if (domain_ids.size() != 7) {
    return Fail("Domain ID count mismatch. Expected 7, got " + std::to_string(domain_ids.size()));
  }

  // 验证边界ID
  const auto boundary_ids = mesh.GetBoundaryIds();
  std::cout << "  Boundary IDs count: " << boundary_ids.size() << std::endl;

  if (boundary_ids.size() != 43) {
    return Fail("Boundary ID count mismatch. Expected 43, got " +
                std::to_string(boundary_ids.size()));
  }

  std::cout << "[PASS] Full mesh read test" << std::endl;
  return 0;
}

int TestMaterials(const std::vector<mpfem::Material>& materials) {
  std::cout << "Testing materials..." << std::endl;

  if (materials.size() != 2) {
    return Fail("Material count mismatch. Expected 2.");
  }

  std::set<std::string> labels;
  for (const auto& material : materials) {
    labels.insert(material.label);
    std::cout << "  Material: " << material.label << " (tag: " << material.tag << ")" << std::endl;
  }

  if (!labels.count("Copper")) {
    return Fail("Material label Copper not found.");
  }
  if (!labels.count("Titanium beta-21S")) {
    return Fail("Material label Titanium beta-21S not found.");
  }

  std::cout << "[PASS] Materials test" << std::endl;
  return 0;
}

int TestProblemDefinition(const mpfem::ProblemDefinition& problem) {
  std::cout << "Testing problem definition..." << std::endl;

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

  std::cout << "[PASS] Problem definition test" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // 设置日志级别（启用 DEBUG 输出）
  mpfem::Logger::Instance().SetLevel(mpfem::LogLevel::kDebug);

  const std::filesystem::path root =
      (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path(".");

  const std::filesystem::path case_path = root / "cases" / "busbar" / "case.xml";

  try {
    MPFEM_INFO("Starting busbar case test");

    // 加载配置
    mpfem::ProblemConfigLoader config_loader;
    const mpfem::ProblemDefinition problem = config_loader.LoadFromXml(case_path.string());

    // 测试网格摘要
    mpfem::ComsolMeshReader mesh_reader;
    const mpfem::ComsolMeshSummary summary = mesh_reader.ReadSummary(problem.mesh_path);
    int result = TestMeshSummary(summary);
    if (result != 0) return result;

    // 测试完整网格读取
    const mpfem::Mesh mesh = mesh_reader.Read(problem.mesh_path);
    result = TestFullMeshRead(mesh);
    if (result != 0) return result;

    // 测试材料
    mpfem::ComsolMaterialReader material_reader;
    const auto materials = material_reader.Read(problem.material_path);
    result = TestMaterials(materials);
    if (result != 0) return result;

    // 测试问题定义
    result = TestProblemDefinition(problem);
    if (result != 0) return result;

    MPFEM_INFO("All busbar case tests passed");
    std::cout << "\n[PASS] All busbar case tests" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    MPFEM_ERROR("Exception: %s", ex.what());
    return Fail(std::string("Unhandled exception: ") + ex.what());
  }
}

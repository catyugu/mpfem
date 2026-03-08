#include "mpfem/solver/coupled_solver.hpp"
#include "mpfem/fem/integrators.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/mesh/mesh.hpp"
#include "mpfem/material/material.hpp"
#include "mpfem/core/logger.hpp"
#include <cmath>

namespace mpfem {

// ============================================================================
// JouleHeatingCoefficient implementation
// ============================================================================

JouleHeatingCoefficient::JouleHeatingCoefficient(const GridFunction* V,
                                                   Coefficient* conductivity)
    : V_(V), conductivity_(conductivity) {}

double JouleHeatingCoefficient::Eval(ElementTransformation& T,
                                      const IntegrationPoint& ip) const {
  // Get electric field magnitude squared: |∇V|²
  Vec3 gradV;
  V_->GetGradient(T, ip, gradV);
  double E_sq = gradV.squaredNorm();

  // Get conductivity
  double sigma = conductivity_ ? conductivity_->Eval(T, ip) : 1.0;

  // Joule heating: Q = σ * |E|² = σ * |∇V|²
  return sigma * E_sq;
}

// ============================================================================
// TemperatureCoefficient implementation
// ============================================================================

double TemperatureCoefficient::Eval(ElementTransformation& T,
                                     const IntegrationPoint& ip) const {
  // Find the element in the mesh
  auto vertices = T.GetElementVertices();
  if (vertices.empty() || !T_) return 293.15;

  const Mesh* mesh = T_->GetFES() ? T_->GetFES()->GetMesh() : nullptr;
  if (!mesh) return 293.15;

  const auto& domain_groups = mesh->DomainElements();
  for (const auto& group : domain_groups) {
    for (int e = 0; e < group.Count(); ++e) {
      auto elem_verts = group.GetElementVertices(e);
      if (elem_verts == vertices) {
        return T_->GetValue(group, e, ip);
      }
    }
  }

  return 293.15;
}

// ============================================================================
// CoupledSolver implementation
// ============================================================================

CoupledSolver::CoupledSolver()
    : electrostatics_(std::make_unique<ElectrostaticsSolver>()),
      heat_(std::make_unique<HeatTransferSolver>()),
      solid_(std::make_unique<SolidMechanicsSolver>()),
      mesh_(nullptr) {}

void CoupledSolver::Setup(const Mesh* mesh,
                           const std::map<int, std::string>& domain_materials,
                           const std::vector<Material>& materials,
                           const ProblemDefinition& problem) {
  mesh_ = mesh;
  domain_materials_ = domain_materials;
  materials_ = materials;

  // Setup individual physics solvers
  electrostatics_->Setup(mesh, domain_materials, materials);
  heat_->Setup(mesh, domain_materials, materials);
  solid_->Setup(mesh, domain_materials, materials);

  // Extract conductivity for Joule heating
  std::map<int, double> conductivity_values;
  for (const auto& [domain_id, material_tag] : domain_materials) {
    for (const auto& mat : materials) {
      if (mat.tag == material_tag) {
        auto it = mat.properties.find("electricconductivity");
        if (it != mat.properties.end()) {
          conductivity_values[domain_id] = it->second.si_value;
        }
        break;
      }
    }
  }
  conductivity_ = CreateMaterialPropertyCoefficient(conductivity_values);

  // Extract thermal expansion coefficient
  std::map<int, double> alpha_values;
  for (const auto& [domain_id, material_tag] : domain_materials) {
    for (const auto& mat : materials) {
      if (mat.tag == material_tag) {
        auto it = mat.properties.find("thermalexpansioncoefficient");
        if (it != mat.properties.end()) {
          alpha_values[domain_id] = it->second.si_value;
        }
        break;
      }
    }
  }
  thermal_expansion_ = CreateMaterialPropertyCoefficient(alpha_values);

  // Apply boundary conditions from problem definition
  ApplyBoundaryConditions(problem);

  MPFEM_INFO("CoupledSolver setup complete");
}

void CoupledSolver::ApplyBoundaryConditions(const ProblemDefinition& problem) {
  // Apply boundary conditions from physics models
  for (const auto& pm : problem.physics) {
    if (pm.kind == PhysicsKind::kElectrostatics) {
      const auto& model = std::get<ElectrostaticsPhysicsModel>(pm.model);
      for (const auto& bc : model.boundary_conditions) {
        if (bc.kind == ElectrostaticsBoundaryKind::kVoltage) {
          // Parse voltage value from expression
          double voltage = 0.0;
          // Try to get from variables
          auto it = problem.variables_si.find(bc.value_expr);
          if (it != problem.variables_si.end()) {
            voltage = it->second;
          } else {
            // Try to parse as number
            try {
              voltage = std::stod(bc.value_expr);
            } catch (...) {
              MPFEM_WARN("Could not parse voltage value: {}", bc.value_expr);
            }
          }

          for (int id : bc.boundary_ids) {
            electrostatics_->SetVoltageBC(id, voltage);
            MPFEM_INFO("Electrostatics: boundary {} voltage = {} V", id, voltage);
          }
        }
      }
    } else if (pm.kind == PhysicsKind::kHeatTransfer) {
      const auto& model = std::get<HeatTransferPhysicsModel>(pm.model);
      for (const auto& bc : model.boundary_conditions) {
        if (bc.kind == HeatBoundaryKind::kConvection) {
          // Parse convection parameters
          double h = 0.0;
          double T_ext = 293.15;

          auto h_it = problem.variables_si.find(bc.value_expr);
          if (h_it != problem.variables_si.end()) {
            h = h_it->second;
          }

          auto t_it = problem.variables_si.find(bc.aux_value_expr);
          if (t_it != problem.variables_si.end()) {
            T_ext = t_it->second;
          } else {
            // Try parsing aux value
            try {
              T_ext = std::stod(bc.aux_value_expr);
            } catch (...) {
              T_ext = 293.15;
            }
          }

          heat_->SetConvectionBC(bc.boundary_ids, h, T_ext);
          MPFEM_INFO("HeatTransfer: convection BC on boundaries, h = {}, T_ext = {} K",
                     h, T_ext);
        }
      }
    } else if (pm.kind == PhysicsKind::kSolidMechanics) {
      const auto& model = std::get<SolidMechanicsPhysicsModel>(pm.model);
      for (const auto& bc : model.boundary_conditions) {
        if (bc.kind == SolidMechanicsBoundaryKind::kFixedConstraint) {
          solid_->SetFixedBC(bc.boundary_ids);
          MPFEM_INFO("SolidMechanics: fixed constraint on boundaries");
        }
      }
    }
  }
}

void CoupledSolver::Solve(int max_iterations, double tolerance) {
  MPFEM_INFO("Starting coupled solve with max {} iterations", max_iterations);

  // For this steady-state weakly-coupled problem, one iteration is typically enough
  // since Joule heating depends on V, and thermal expansion depends on T

  // Iteration 1: Solve electrostatics
  MPFEM_INFO("=== Iteration 1: Solving Electrostatics ===");
  electrostatics_->Assemble();
  electrostatics_->Solve();

  // Setup Joule heating coefficient using the solved potential
  joule_heat_ = std::make_unique<JouleHeatingCoefficient>(
      electrostatics_->GetSolution(), conductivity_.get());

  // Set heat source
  heat_->SetHeatSource(joule_heat_.get());

  // Iteration 1: Solve heat transfer
  MPFEM_INFO("=== Iteration 1: Solving Heat Transfer ===");
  heat_->Assemble();
  heat_->Solve();

  // Setup thermal expansion using the solved temperature
  temp_coeff_ = std::make_unique<TemperatureCoefficient>(heat_->GetSolution());

  // Set thermal expansion for solid mechanics
  solid_->SetThermalExpansion(thermal_expansion_.get(), temp_coeff_.get(), 293.15);

  // Iteration 1: Solve solid mechanics
  MPFEM_INFO("=== Iteration 1: Solving Solid Mechanics ===");
  solid_->Assemble();
  solid_->Solve();

  // For strongly coupled problems, additional iterations would be needed
  // For this case (weakly coupled: V→Q→T→ε_th→u), one pass is sufficient

  MPFEM_INFO("Coupled solve complete");

  // Log results
  auto* V = GetElectricPotential();
  auto* T = GetTemperature();
  auto* u = GetDisplacement();

  MPFEM_INFO("Results:");
  MPFEM_INFO("  Electric potential: [{}, {}] V", V->Min(), V->Max());
  MPFEM_INFO("  Temperature: [{}, {}] K", T->Min(), T->Max());
  MPFEM_INFO("  Displacement magnitude: {}", u->Data().norm());
}

}  // namespace mpfem

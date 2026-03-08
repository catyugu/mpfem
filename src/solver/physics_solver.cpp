#include "mpfem/solver/physics_solver.hpp"
#include "mpfem/fem/integrators.hpp"
#include "mpfem/fem/fe.hpp"
#include "mpfem/mesh/mesh.hpp"
#include "mpfem/material/material.hpp"
#include "mpfem/core/logger.hpp"
#include <algorithm>

namespace mpfem {

// ============================================================================
// ElectrostaticsSolver implementation
// ============================================================================

ElectrostaticsSolver::ElectrostaticsSolver()
    : mesh_(nullptr), solver_(SolverFactory::Create(SolverFactory::SolverType::kSparseLU)) {}

void ElectrostaticsSolver::Setup(const Mesh* mesh,
                                  const std::map<int, std::string>& domain_materials,
                                  const std::vector<Material>& materials) {
  mesh_ = mesh;

  // Create FE collection and space
  fec_ = std::make_unique<H1_FECollection>(1, mesh->SpaceDim());
  fes_ = std::make_unique<FiniteElementSpace>(mesh, fec_.get(), 1);

  // Create solution grid function
  V_ = std::make_unique<GridFunction>(fes_.get());

  // Create forms
  a_ = std::make_unique<BilinearForm>(fes_.get());
  b_ = std::make_unique<LinearForm>(fes_.get());

  // Extract conductivity from materials
  std::map<int, double> conductivity_values;
  for (const auto& [domain_id, material_tag] : domain_materials) {
    for (const auto& mat : materials) {
      if (mat.tag == material_tag) {
        auto it = mat.properties.find("electricconductivity");
        if (it != mat.properties.end()) {
          conductivity_values[domain_id] = it->second.si_value;
          MPFEM_INFO("Domain %d: conductivity = %g S/m", domain_id, it->second.si_value);
        }
        break;
      }
    }
  }

  // Create conductivity coefficient
  conductivity_ = CreateMaterialPropertyCoefficient(conductivity_values);

  // Initialize boundary marker
  auto bdr_ids = mesh->GetBoundaryIds();
  int max_bdr_id = bdr_ids.empty() ? 0 : *std::max_element(bdr_ids.begin(), bdr_ids.end());
  ess_bdr_marker_.resize(max_bdr_id, 0);

  MPFEM_INFO("ElectrostaticsSolver setup complete: %d DOFs", fes_->GetNDofs());
}

void ElectrostaticsSolver::SetVoltageBC(int boundary_id, double voltage) {
  voltage_bc_[boundary_id] = voltage;
  if (boundary_id > 0 && boundary_id <= static_cast<int>(ess_bdr_marker_.size())) {
    ess_bdr_marker_[boundary_id - 1] = 1;
  }
}

void ElectrostaticsSolver::SetGroundBC(const std::vector<int>& boundary_ids) {
  for (int id : boundary_ids) {
    SetVoltageBC(id, 0.0);
  }
}

void ElectrostaticsSolver::Assemble() {
  // Add diffusion integrator with conductivity
  a_->AddDomainIntegrator(std::make_unique<DiffusionIntegrator>(conductivity_.get()));
  a_->Assemble();

  // RHS is zero for electrostatics (no source terms)
  b_->Assemble();
}

void ElectrostaticsSolver::Solve() {
  // Get essential DOFs
  std::vector<int> ess_dofs;
  fes_->GetEssentialTrueDofs(ess_bdr_marker_, ess_dofs);

  // Set Dirichlet BC values
  VectorXd x(fes_->GetVSize());
  x.setZero();

  // Apply voltage BC values
  const auto& bdr_groups = mesh_->BoundaryElements();
  ElementTransformation T;
  T.SetMesh(mesh_);

  for (size_t g = 0; g < bdr_groups.size(); ++g) {
    const auto& group = bdr_groups[g];
    for (int e = 0; e < group.Count(); ++e) {
      int bdr_id = group.entity_ids[e];
      auto it = voltage_bc_.find(bdr_id);
      if (it == voltage_bc_.end()) continue;

      double voltage = it->second;
      auto verts = group.GetElementVertices(e);

      for (int v : verts) {
        x(v) = voltage;
      }
    }
  }

  // Form linear system
  SparseMatrix A;
  VectorXd X, B;
  VectorXd b = b_->GetVector();
  a_->FormLinearSystem(ess_dofs, x, b, A, X, B);

  // Solve
  solver_->SetOperator(A);
  solver_->Mult(B, X);

  // Copy solution back
  int n_free = X.size();
  std::vector<bool> is_essential(fes_->GetVSize(), false);
  for (int dof : ess_dofs) {
    is_essential[dof] = true;
  }

  VectorXd& V_data = V_->Data();
  int free_idx = 0;
  for (int i = 0; i < fes_->GetVSize(); ++i) {
    if (is_essential[i]) {
      V_data(i) = x(i);
    } else {
      V_data(i) = X(free_idx++);
    }
  }

  MPFEM_INFO("Electrostatics solved: V_min = %f, V_max = %f", V_->Min(), V_->Max());
}

std::unique_ptr<VectorCoefficient> ElectrostaticsSolver::GetElectricField() const {
  return std::make_unique<GridFunctionGradientCoefficient>(V_.get());
}

// ============================================================================
// HeatTransferSolver implementation
// ============================================================================

HeatTransferSolver::HeatTransferSolver()
    : mesh_(nullptr), solver_(SolverFactory::Create(SolverFactory::SolverType::kSparseLU)) {}

void HeatTransferSolver::Setup(const Mesh* mesh,
                                const std::map<int, std::string>& domain_materials,
                                const std::vector<Material>& materials) {
  mesh_ = mesh;

  fec_ = std::make_unique<H1_FECollection>(1, mesh->SpaceDim());
  fes_ = std::make_unique<FiniteElementSpace>(mesh, fec_.get(), 1);

  T_ = std::make_unique<GridFunction>(fes_.get());
  T_->SetConstant(293.15);  // Initial temperature

  a_ = std::make_unique<BilinearForm>(fes_.get());
  b_ = std::make_unique<LinearForm>(fes_.get());

  // Extract thermal conductivity from materials
  std::map<int, double> k_values;
  for (const auto& [domain_id, material_tag] : domain_materials) {
    for (const auto& mat : materials) {
      if (mat.tag == material_tag) {
        auto it = mat.properties.find("thermalconductivity");
        if (it != mat.properties.end()) {
          k_values[domain_id] = it->second.si_value;
          MPFEM_INFO("Domain %d: thermal conductivity = %g W/(m·K)", domain_id, it->second.si_value);
        }
        break;
      }
    }
  }

  thermal_conductivity_ = CreateMaterialPropertyCoefficient(k_values);

  auto bdr_ids = mesh->GetBoundaryIds();
  int max_bdr_id = bdr_ids.empty() ? 0 : *std::max_element(bdr_ids.begin(), bdr_ids.end());
  ess_bdr_marker_.resize(max_bdr_id, 0);

  MPFEM_INFO("HeatTransferSolver setup complete: %d DOFs", fes_->GetNDofs());
}

void HeatTransferSolver::SetTemperatureBC(int boundary_id, double temperature) {
  // TODO: Implement temperature BC
  if (boundary_id > 0 && boundary_id <= static_cast<int>(ess_bdr_marker_.size())) {
    ess_bdr_marker_[boundary_id - 1] = 1;
  }
}

void HeatTransferSolver::SetConvectionBC(const std::vector<int>& boundary_ids,
                                          double h, double T_ext) {
  convection_bdr_ = boundary_ids;
  h_ = h;
  T_ext_ = T_ext;
}

void HeatTransferSolver::SetHeatInsulationBC(const std::vector<int>& boundary_ids) {
  // Heat insulation = zero heat flux = natural BC (do nothing)
  (void)boundary_ids;
}

void HeatTransferSolver::SetHeatSource(Coefficient* Q) {
  Q_source_ = Q;
}

void HeatTransferSolver::Assemble() {
  // Add diffusion integrator
  a_->AddDomainIntegrator(std::make_unique<DiffusionIntegrator>(thermal_conductivity_.get()));

  // Add convection boundary condition
  if (!convection_bdr_.empty() && h_ > 0) {
    auto h_coeff = std::make_unique<ConstantCoefficient>(h_);
    auto T_ext_coeff = std::make_unique<ConstantCoefficient>(T_ext_);

    a_->AddBoundaryIntegrator(
        std::make_unique<ConvectionIntegrator>(h_coeff.get(), T_ext_coeff.get()),
        convection_bdr_);

    b_->AddBoundaryIntegrator(
        std::make_unique<ConvectionRHSIntegrator>(h_coeff.get(), T_ext_coeff.get()),
        convection_bdr_);

    // Store the coefficients to prevent deletion
    // (This is a workaround; proper ownership management needed)
  }

  a_->Assemble();

  // Add heat source
  if (Q_source_) {
    // Get all domain IDs
    std::vector<int> all_domains = mesh_->GetDomainIds();
    b_->AddDomainIntegrator(std::make_unique<DomainLFIntegrator>(Q_source_), all_domains);
  }

  b_->Assemble();
}

void HeatTransferSolver::Solve() {
  std::vector<int> ess_dofs;
  fes_->GetEssentialTrueDofs(ess_bdr_marker_, ess_dofs);

  VectorXd x = T_->Data();

  SparseMatrix A;
  VectorXd X, B;
  VectorXd b = b_->GetVector();
  a_->FormLinearSystem(ess_dofs, x, b, A, X, B);

  solver_->SetOperator(A);
  solver_->Mult(B, X);

  // Copy solution back
  VectorXd& T_data = T_->Data();
  std::vector<bool> is_essential(fes_->GetVSize(), false);
  for (int dof : ess_dofs) {
    is_essential[dof] = true;
  }

  int free_idx = 0;
  for (int i = 0; i < fes_->GetVSize(); ++i) {
    if (!is_essential[i]) {
      T_data(i) = X(free_idx++);
    }
  }

  MPFEM_INFO("HeatTransfer solved: T_min = %g K, T_max = %g K", T_->Min(), T_->Max());
}

// ============================================================================
// SolidMechanicsSolver implementation
// ============================================================================

SolidMechanicsSolver::SolidMechanicsSolver()
    : mesh_(nullptr), solver_(SolverFactory::Create(SolverFactory::SolverType::kSparseLU)) {}

void SolidMechanicsSolver::Setup(const Mesh* mesh,
                                  const std::map<int, std::string>& domain_materials,
                                  const std::vector<Material>& materials) {
  mesh_ = mesh;

  fec_ = std::make_unique<H1_FECollection>(1, mesh->SpaceDim());
  fes_ = std::make_unique<FiniteElementSpace>(mesh, fec_.get(), 3);  // 3D displacement

  u_ = std::make_unique<GridFunction>(fes_.get());
  u_->SetZero();

  a_ = std::make_unique<BilinearForm>(fes_.get());
  b_ = std::make_unique<LinearForm>(fes_.get());

  // Extract elastic properties from materials
  std::map<int, double> E_values, nu_values;
  for (const auto& [domain_id, material_tag] : domain_materials) {
    for (const auto& mat : materials) {
      if (mat.tag == material_tag) {
        auto E_it = mat.properties.find("E");
        if (E_it != mat.properties.end()) {
          E_values[domain_id] = E_it->second.si_value;
        }
        auto nu_it = mat.properties.find("nu");
        if (nu_it != mat.properties.end()) {
          nu_values[domain_id] = nu_it->second.si_value;
        }
        MPFEM_INFO("Domain %d: E = %g Pa, nu = %g", domain_id,
                   E_it != mat.properties.end() ? E_it->second.si_value : 0,
                   nu_it != mat.properties.end() ? nu_it->second.si_value : 0);
        break;
      }
    }
  }

  // Compute Lamé parameters
  std::map<int, double> lambda_values, mu_values;
  for (const auto& [domain_id, E] : E_values) {
    double nu = nu_values.count(domain_id) ? nu_values[domain_id] : 0.3;
    lambda_values[domain_id] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    mu_values[domain_id] = E / (2.0 * (1.0 + nu));
  }

  lambda_ = CreateMaterialPropertyCoefficient(lambda_values);
  mu_ = CreateMaterialPropertyCoefficient(mu_values);

  auto bdr_ids = mesh->GetBoundaryIds();
  int max_bdr_id = bdr_ids.empty() ? 0 : *std::max_element(bdr_ids.begin(), bdr_ids.end());
  ess_bdr_marker_.resize(max_bdr_id, 0);

  MPFEM_INFO("SolidMechanicsSolver setup complete: %d DOFs", fes_->GetNDofs());
}

void SolidMechanicsSolver::SetFixedBC(const std::vector<int>& boundary_ids) {
  for (int id : boundary_ids) {
    if (id > 0 && id <= static_cast<int>(ess_bdr_marker_.size())) {
      ess_bdr_marker_[id - 1] = 1;
    }
    displacement_bc_[id] = Vec3::Zero();
  }
}

void SolidMechanicsSolver::SetDisplacementBC(int boundary_id, const Vec3& displacement) {
  if (boundary_id > 0 && boundary_id <= static_cast<int>(ess_bdr_marker_.size())) {
    ess_bdr_marker_[boundary_id - 1] = 1;
  }
  displacement_bc_[boundary_id] = displacement;
}

void SolidMechanicsSolver::SetThermalExpansion(Coefficient* alpha, Coefficient* T, double T_ref) {
  alpha_ = alpha;
  T_ = T;
  T_ref_ = T_ref;
}

void SolidMechanicsSolver::Assemble() {
  // Add elasticity integrator
  a_->AddDomainIntegrator(
      std::make_unique<ElasticityIntegrator>(lambda_.get(), mu_.get()));
  a_->Assemble();

  // Add thermal expansion as initial strain
  if (alpha_ && T_) {
    b_->AddDomainIntegrator(
        std::make_unique<ThermalExpansionIntegrator>(alpha_, T_, T_ref_,
                                                      lambda_.get(), mu_.get()));
  }

  b_->Assemble();
}

void SolidMechanicsSolver::Solve() {
  std::vector<int> ess_dofs;
  fes_->GetEssentialTrueDofs(ess_bdr_marker_, ess_dofs);

  // Set Dirichlet BC values
  VectorXd x(fes_->GetVSize());
  x.setZero();

  const auto& bdr_groups = mesh_->BoundaryElements();
  ElementTransformation T;
  T.SetMesh(mesh_);

  for (size_t g = 0; g < bdr_groups.size(); ++g) {
    const auto& group = bdr_groups[g];
    for (int e = 0; e < group.Count(); ++e) {
      int bdr_id = group.entity_ids[e];
      auto it = displacement_bc_.find(bdr_id);
      if (it == displacement_bc_.end()) continue;

      const Vec3& disp = it->second;
      auto verts = group.GetElementVertices(e);

      for (int v : verts) {
        for (int c = 0; c < 3; ++c) {
          x(v * 3 + c) = disp(c);
        }
      }
    }
  }

  SparseMatrix A;
  VectorXd X, B;
  VectorXd b = b_->GetVector();
  a_->FormLinearSystem(ess_dofs, x, b, A, X, B);

  solver_->SetOperator(A);
  solver_->Mult(B, X);

  // Copy solution back
  VectorXd& u_data = u_->Data();
  std::vector<bool> is_essential(fes_->GetVSize(), false);
  for (int dof : ess_dofs) {
    is_essential[dof] = true;
  }

  int free_idx = 0;
  for (int i = 0; i < fes_->GetVSize(); ++i) {
    if (!is_essential[i]) {
      u_data(i) = X(free_idx++);
    }
  }

  MPFEM_INFO("SolidMechanics solved: displacement magnitude range = [%g, %g]",
             u_data.head(fes_->GetNDofs()).norm(),
             u_data.tail(fes_->GetNDofs() * 2).norm());
}

}  // namespace mpfem

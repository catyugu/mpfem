#pragma once

#include <memory>
#include <vector>

#include "mpfem/core/types.hpp"
#include "mpfem/core/problem_definition.hpp"
#include "mpfem/material/material.hpp"
#include "mpfem/fem/fe_space.hpp"
#include "mpfem/fem/grid_function.hpp"
#include "mpfem/fem/coefficient.hpp"
#include "mpfem/fem/bilinear_form.hpp"
#include "mpfem/solver/linear_solver.hpp"

namespace mpfem {

// Forward declarations
class Mesh;

// ============================================================================
// PhysicsSolverBase - 物理场求解器基类
// ============================================================================

class PhysicsSolverBase {
 public:
  virtual ~PhysicsSolverBase() = default;

  // Setup the solver with mesh, materials, and boundary conditions
  virtual void Setup(const Mesh* mesh,
                     const std::map<int, std::string>& domain_materials,
                     const std::vector<Material>& materials) = 0;

  // Assemble the system matrix and vector
  virtual void Assemble() = 0;

  // Solve the system
  virtual void Solve() = 0;

  // Get the solution
  virtual GridFunction* GetSolution() = 0;

  // Get the FE space
  virtual FiniteElementSpace* GetFESpace() = 0;
};

// ============================================================================
// ElectrostaticsSolver - 静电场求解器
// 方程: -∇·(σ ∇V) = 0
// ============================================================================

class ElectrostaticsSolver : public PhysicsSolverBase {
 public:
  ElectrostaticsSolver();
  ~ElectrostaticsSolver() override = default;

  void Setup(const Mesh* mesh,
             const std::map<int, std::string>& domain_materials,
             const std::vector<Material>& materials) override;

  void Assemble() override;
  void Solve() override;

  GridFunction* GetSolution() override { return V_.get(); }
  FiniteElementSpace* GetFESpace() override { return fes_.get(); }

  // Set boundary conditions
  void SetVoltageBC(int boundary_id, double voltage);
  void SetGroundBC(const std::vector<int>& boundary_ids);

  // Get electric field (gradient of potential)
  std::unique_ptr<VectorCoefficient> GetElectricField() const;

  // Get essential DOF marker
  const std::vector<int>& GetEssentialBoundaryMarker() const {
    return ess_bdr_marker_;
  }

 private:
  const Mesh* mesh_;
  std::unique_ptr<FiniteElementSpace> fes_;
  std::unique_ptr<H1_FECollection> fec_;
  std::unique_ptr<GridFunction> V_;

  std::unique_ptr<BilinearForm> a_;
  std::unique_ptr<LinearForm> b_;
  std::unique_ptr<LinearSolver> solver_;

  // Boundary conditions
  std::vector<int> ess_bdr_marker_;  // 1 = essential BC
  std::map<int, double> voltage_bc_;  // boundary_id -> voltage value

  // Material properties
  std::unique_ptr<PWConstCoefficient> conductivity_;
};

// ============================================================================
// HeatTransferSolver - 热传导求解器
// 方程: -∇·(k ∇T) = Q (焦耳热源)
// ============================================================================

class HeatTransferSolver : public PhysicsSolverBase {
 public:
  HeatTransferSolver();
  ~HeatTransferSolver() override = default;

  void Setup(const Mesh* mesh,
             const std::map<int, std::string>& domain_materials,
             const std::vector<Material>& materials) override;

  void Assemble() override;
  void Solve() override;

  GridFunction* GetSolution() override { return T_.get(); }
  FiniteElementSpace* GetFESpace() override { return fes_.get(); }

  // Set boundary conditions
  void SetTemperatureBC(int boundary_id, double temperature);
  void SetConvectionBC(const std::vector<int>& boundary_ids,
                       double h, double T_ext);
  void SetHeatInsulationBC(const std::vector<int>& boundary_ids);

  // Set heat source
  void SetHeatSource(Coefficient* Q);

  // Get essential DOF marker
  const std::vector<int>& GetEssentialBoundaryMarker() const {
    return ess_bdr_marker_;
  }

 private:
  const Mesh* mesh_;
  std::unique_ptr<FiniteElementSpace> fes_;
  std::unique_ptr<H1_FECollection> fec_;
  std::unique_ptr<GridFunction> T_;

  std::unique_ptr<BilinearForm> a_;
  std::unique_ptr<LinearForm> b_;
  std::unique_ptr<LinearSolver> solver_;

  // Boundary conditions
  std::vector<int> ess_bdr_marker_;
  std::vector<int> convection_bdr_;  // Boundaries with convection
  double h_ = 0.0;                   // Heat transfer coefficient
  double T_ext_ = 293.15;            // External temperature

  // Heat source
  Coefficient* Q_source_ = nullptr;

  // Material properties
  std::unique_ptr<PWConstCoefficient> thermal_conductivity_;
};

// ============================================================================
// SolidMechanicsSolver - 固体力学求解器
// 方程: ∇·σ + f = 0
// ============================================================================

class SolidMechanicsSolver : public PhysicsSolverBase {
 public:
  SolidMechanicsSolver();
  ~SolidMechanicsSolver() override = default;

  void Setup(const Mesh* mesh,
             const std::map<int, std::string>& domain_materials,
             const std::vector<Material>& materials) override;

  void Assemble() override;
  void Solve() override;

  GridFunction* GetSolution() override { return u_.get(); }
  FiniteElementSpace* GetFESpace() override { return fes_.get(); }

  // Set boundary conditions
  void SetFixedBC(const std::vector<int>& boundary_ids);
  void SetDisplacementBC(int boundary_id, const Vec3& displacement);

  // Set thermal expansion (from temperature field)
  void SetThermalExpansion(Coefficient* alpha, Coefficient* T, double T_ref);

  // Get essential DOF marker
  const std::vector<int>& GetEssentialBoundaryMarker() const {
    return ess_bdr_marker_;
  }

 private:
  const Mesh* mesh_;
  std::unique_ptr<FiniteElementSpace> fes_;
  std::unique_ptr<H1_FECollection> fec_;
  std::unique_ptr<GridFunction> u_;

  std::unique_ptr<BilinearForm> a_;
  std::unique_ptr<LinearForm> b_;
  std::unique_ptr<LinearSolver> solver_;

  // Boundary conditions
  std::vector<int> ess_bdr_marker_;
  std::map<int, Vec3> displacement_bc_;

  // Material properties (Lamé parameters)
  std::unique_ptr<PWConstCoefficient> lambda_;
  std::unique_ptr<PWConstCoefficient> mu_;

  // Thermal expansion
  Coefficient* alpha_ = nullptr;
  Coefficient* T_ = nullptr;
  double T_ref_ = 293.15;
};

}  // namespace mpfem

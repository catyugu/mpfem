#pragma once

#include <memory>
#include <vector>

#include "mpfem/core/problem_definition.hpp"
#include "mpfem/solver/physics_solver.hpp"
#include "mpfem/fem/coefficient.hpp"
#include "mpfem/fem/grid_function.hpp"

namespace mpfem {

// ============================================================================
// JouleHeatingCoefficient - 焦耳热系数
// Q = σ * |E|² = σ * |∇V|²
// ============================================================================

class JouleHeatingCoefficient : public Coefficient {
 public:
  JouleHeatingCoefficient(const GridFunction* V, Coefficient* conductivity);

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override;

 private:
  const GridFunction* V_;
  Coefficient* conductivity_;
};

// ============================================================================
// TemperatureCoefficient - 温度系数（包装 GridFunction）
// ============================================================================

class TemperatureCoefficient : public Coefficient {
 public:
  explicit TemperatureCoefficient(const GridFunction* T) : T_(T) {}

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override;

 private:
  const GridFunction* T_;
};

// ============================================================================
// ThermalExpansionCoefficient - 热膨胀系数包装
// ============================================================================

class ThermalExpansionCoefficient : public Coefficient {
 public:
  ThermalExpansionCoefficient(Coefficient* base_coeff)
      : base_coeff_(base_coeff) {}

  double Eval(ElementTransformation& T,
              const IntegrationPoint& ip) const override {
    return base_coeff_ ? base_coeff_->Eval(T, ip) : 0.0;
  }

 private:
  Coefficient* base_coeff_;
};

// ============================================================================
// CoupledSolver - 耦合物理场求解器
// 顺序耦合迭代求解：
// 1. 求解静电场 → V
// 2. 计算焦耳热 → Q
// 3. 求解热传导 → T
// 4. 计算热膨胀 → ε_th
// 5. 求解固体力学 → u
// ============================================================================

class CoupledSolver {
 public:
  CoupledSolver();
  ~CoupledSolver() = default;

  // Setup all physics solvers
  void Setup(const Mesh* mesh,
             const std::map<int, std::string>& domain_materials,
             const std::vector<Material>& materials,
             const ProblemDefinition& problem);

  // Solve the coupled system
  void Solve(int max_iterations = 10, double tolerance = 1e-6);

  // Get results
  GridFunction* GetElectricPotential() { return electrostatics_->GetSolution(); }
  GridFunction* GetTemperature() { return heat_->GetSolution(); }
  GridFunction* GetDisplacement() { return solid_->GetSolution(); }

  // Get individual solvers
  ElectrostaticsSolver* GetElectrostaticsSolver() { return electrostatics_.get(); }
  HeatTransferSolver* GetHeatSolver() { return heat_.get(); }
  SolidMechanicsSolver* GetSolidMechanicsSolver() { return solid_.get(); }

 private:
  void ApplyBoundaryConditions(const ProblemDefinition& problem);

  // Physics solvers
  std::unique_ptr<ElectrostaticsSolver> electrostatics_;
  std::unique_ptr<HeatTransferSolver> heat_;
  std::unique_ptr<SolidMechanicsSolver> solid_;

  // Coupling coefficients
  std::unique_ptr<JouleHeatingCoefficient> joule_heat_;
  std::unique_ptr<TemperatureCoefficient> temp_coeff_;

  // Material properties
  std::unique_ptr<PWConstCoefficient> conductivity_;
  std::unique_ptr<PWConstCoefficient> thermal_expansion_;

  const Mesh* mesh_;
  std::map<int, std::string> domain_materials_;
  std::vector<Material> materials_;
};

}  // namespace mpfem

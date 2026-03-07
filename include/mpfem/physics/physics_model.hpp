#pragma once

#include <string>
#include <variant>
#include <vector>

#include "mpfem/core/types.hpp"

namespace mpfem {

struct DomainScope {
  // Empty means this model applies to all domains.
  std::vector<int> domain_ids;

  [[nodiscard]] bool AppliesToAllDomains() const { return domain_ids.empty(); }
};

enum class ElectrostaticsBoundaryKind {
  kVoltage,
  kNormalCurrentDensity,
  kElectricInsulation
};

enum class ElectrostaticsSourceKind {
  kVolumeChargeDensity,
  kVolumeCurrentSource
};

struct ElectrostaticsBoundaryCondition {
  ElectrostaticsBoundaryKind kind = ElectrostaticsBoundaryKind::kVoltage;
  std::vector<int> boundary_ids;
  std::string value_expr;
};

struct ElectrostaticsSourceTerm {
  ElectrostaticsSourceKind kind = ElectrostaticsSourceKind::kVolumeChargeDensity;
  std::vector<int> domain_ids;
  std::string value_expr;
};

struct ElectrostaticsPhysicsModel {
  std::vector<ElectrostaticsBoundaryCondition> boundary_conditions;
  std::vector<ElectrostaticsSourceTerm> sources;
};

enum class HeatBoundaryKind {
  kTemperature,
  kHeatFlux,
  kConvection,
  kThermalInsulation
};

enum class HeatSourceKind {
  kVolumetricHeatSource
};

struct HeatBoundaryCondition {
  HeatBoundaryKind kind = HeatBoundaryKind::kTemperature;
  std::vector<int> boundary_ids;
  std::string value_expr;
  std::string aux_value_expr;
};

struct HeatSourceTerm {
  HeatSourceKind kind = HeatSourceKind::kVolumetricHeatSource;
  std::vector<int> domain_ids;
  std::string value_expr;
};

struct HeatTransferPhysicsModel {
  std::vector<HeatBoundaryCondition> boundary_conditions;
  std::vector<HeatSourceTerm> sources;
};

enum class SolidMechanicsBoundaryKind {
  kDisplacement,
  kTraction,
  kFixedConstraint,
  kFreeBoundary
};

enum class SolidMechanicsSourceKind {
  kBodyForce,
  kInitialStrain
};

struct SolidMechanicsBoundaryCondition {
  SolidMechanicsBoundaryKind kind = SolidMechanicsBoundaryKind::kDisplacement;
  std::vector<int> boundary_ids;
  std::string value_expr;
};

struct SolidMechanicsSourceTerm {
  SolidMechanicsSourceKind kind = SolidMechanicsSourceKind::kBodyForce;
  std::vector<int> domain_ids;
  std::string value_expr;
};

struct SolidMechanicsPhysicsModel {
  std::vector<SolidMechanicsBoundaryCondition> boundary_conditions;
  std::vector<SolidMechanicsSourceTerm> sources;
};

struct PhysicsModel {
  PhysicsKind kind = PhysicsKind::kElectrostatics;
  DomainScope scope;

  std::variant<
      ElectrostaticsPhysicsModel,
      HeatTransferPhysicsModel,
      SolidMechanicsPhysicsModel>
      model;
};

}  // namespace mpfem

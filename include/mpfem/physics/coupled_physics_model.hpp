#pragma once

#include <string>
#include <vector>

#include "mpfem/core/types.hpp"
#include "mpfem/physics/physics_model.hpp"

namespace mpfem {

enum class CoupledPhysicsKind {
  kJouleHeating,
  kThermalExpansion,
  kCustom
};

inline CoupledPhysicsKind CoupledPhysicsKindFromString(const std::string& value) {
  if (value == "joule_heating") {
    return CoupledPhysicsKind::kJouleHeating;
  }
  if (value == "thermal_expansion") {
    return CoupledPhysicsKind::kThermalExpansion;
  }
  return CoupledPhysicsKind::kCustom;
}

struct CoupledPhysicsModel {
  std::string name;
  CoupledPhysicsKind kind = CoupledPhysicsKind::kCustom;
  std::vector<PhysicsKind> sub_physics;

  // Empty means coupled model applies to all domains allowed by sub-physics.
  DomainScope scope;
};

}  // namespace mpfem

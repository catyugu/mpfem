#pragma once

#include <stdexcept>
#include <string>

namespace mpfem {

enum class FESpaceKind {
  kH1,
  kND,
  kRT,
  kL2
};

enum class StudyKind {
  kSteady,
  kTransient,
  kEigen
};

enum class PhysicsKind {
  kElectrostatics,
  kHeatTransfer,
  kSolidMechanics
};

inline PhysicsKind PhysicsKindFromString(const std::string& value) {
  if (value == "electrostatics") {
    return PhysicsKind::kElectrostatics;
  }
  if (value == "heat_transfer") {
    return PhysicsKind::kHeatTransfer;
  }
  if (value == "solid_mechanics") {
    return PhysicsKind::kSolidMechanics;
  }
  throw std::runtime_error("Unsupported physics kind: " + value);
}

inline std::string ToString(const PhysicsKind kind) {
  switch (kind) {
    case PhysicsKind::kElectrostatics:
      return "electrostatics";
    case PhysicsKind::kHeatTransfer:
      return "heat_transfer";
    case PhysicsKind::kSolidMechanics:
      return "solid_mechanics";
  }
  return "unknown";
}

}  // namespace mpfem

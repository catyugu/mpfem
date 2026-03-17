#ifndef MPFEM_FIELD_KIND_HPP
#define MPFEM_FIELD_KIND_HPP

#include <string>

namespace mpfem {

/**
 * @brief Enumeration of physics field types.
 */
enum class FieldKind {
    ElectricPotential,
    Temperature,
    Displacement,
    Unknown
};

/**
 * @brief Convert FieldKind to string.
 */
inline std::string fieldKindToString(FieldKind kind) {
    switch (kind) {
        case FieldKind::ElectricPotential: return "electric_potential";
        case FieldKind::Temperature: return "temperature";
        case FieldKind::Displacement: return "displacement";
        default: return "unknown";
    }
}

/**
 * @brief Convert string to FieldKind.
 */
inline FieldKind stringToFieldKind(const std::string& str) {
    if (str == "electric_potential" || str == "electrostatics") {
        return FieldKind::ElectricPotential;
    }
    if (str == "temperature" || str == "heat_transfer") {
        return FieldKind::Temperature;
    }
    if (str == "displacement" || str == "solid_mechanics") {
        return FieldKind::Displacement;
    }
    return FieldKind::Unknown;
}

/**
 * @brief Enumeration of coupling types.
 */
enum class CouplingKind {
    JouleHeating,
    ThermalExpansion,
    Unknown
};

/**
 * @brief Convert CouplingKind to string.
 */
inline std::string couplingKindToString(CouplingKind kind) {
    switch (kind) {
        case CouplingKind::JouleHeating: return "joule_heating";
        case CouplingKind::ThermalExpansion: return "thermal_expansion";
        default: return "unknown";
    }
}

/**
 * @brief Convert string to CouplingKind.
 */
inline CouplingKind stringToCouplingKind(const std::string& str) {
    if (str == "joule_heating" || str == "joule_heat") {
        return CouplingKind::JouleHeating;
    }
    if (str == "thermal_expansion") {
        return CouplingKind::ThermalExpansion;
    }
    return CouplingKind::Unknown;
}

}  // namespace mpfem

#endif  // MPFEM_FIELD_KIND_HPP

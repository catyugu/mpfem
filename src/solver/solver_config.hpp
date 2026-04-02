#ifndef MPFEM_SOLVER_CONFIG_HPP
#define MPFEM_SOLVER_CONFIG_HPP

#include "core/types.hpp"
#include <map>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace mpfem {

// =============================================================================
// Solver Type Enumerations
// =============================================================================

enum class LinearSolverType {
    // Eigen solvers (always available)
    Eigen_SparseLU,       ///< Direct LU factorization
    Eigen_CG,             ///< Conjugate Gradient
    Eigen_DGMRES,         ///< DGMRES / FGMRES style Krylov solver

    // MKL PARDISO (conditionally available)
    MKL_Pardiso,          ///< MKL PARDISO direct solver

    // SuiteSparse UMFPACK (conditionally available)
    UMFPACK_LU,           ///< SuiteSparse UMFPACK direct solver
};

enum class PreconditionerType {
    None,
    Diagonal,
    ICC,
    ILU,
    AdditiveSchwarz,
    AMG,
    GaussSeidel,
};

enum class SolverNodeRole {
    Preconditioner,
    LocalSolver,
    CoarseSolver,
    Smoother,
};

// =============================================================================
// Solver Metadata
// =============================================================================

struct LinearSolverMeta {
    LinearSolverType type;
    std::string_view name;
    std::string_view description;
    bool isIterative;
    bool requiresSPD;
    bool isAvailable;
};

// =============================================================================
// Solver Registry
// =============================================================================

namespace detail {

inline constexpr bool isMKLAvailable() {
#ifdef MPFEM_USE_MKL
    return true;
#else
    return false;
#endif
}

inline constexpr bool isSuiteSparseAvailable() {
#ifdef MPFEM_USE_SUITESPARSE
    return true;
#else
    return false;
#endif
}

inline constexpr LinearSolverMeta linearSolverRegistry[] = {
    // Eigen solvers (always available)
    {LinearSolverType::Eigen_SparseLU, "SparseLU", "Eigen SparseLU direct solver", false, false, true},
    {LinearSolverType::Eigen_CG,       "CG",       "Eigen Conjugate Gradient solver", true, true, true},
    {LinearSolverType::Eigen_DGMRES,   "FGMRES",   "Eigen DGMRES/FGMRES solver", true, false, true},

    // MKL PARDISO
    {LinearSolverType::MKL_Pardiso, "Pardiso", "MKL PARDISO direct solver", false, false, isMKLAvailable()},

    // SuiteSparse UMFPACK
    {LinearSolverType::UMFPACK_LU, "UMFPACK", "SuiteSparse UMFPACK direct solver", false, false, isSuiteSparseAvailable()},
};

inline constexpr size_t linearSolverRegistrySize = sizeof(linearSolverRegistry) / sizeof(LinearSolverMeta);

}  // namespace detail

// =============================================================================
// Utility Functions
// =============================================================================

inline const LinearSolverMeta& getLinearSolverMeta(LinearSolverType type) {
    for (size_t i = 0; i < detail::linearSolverRegistrySize; ++i) {
        if (detail::linearSolverRegistry[i].type == type) {
            return detail::linearSolverRegistry[i];
        }
    }
    throw std::runtime_error("Unknown linear solver type");
}

inline std::string_view linearSolverTypeName(LinearSolverType type) {
    return getLinearSolverMeta(type).name;
}

inline bool isLinearSolverAvailable(LinearSolverType type) {
    return getLinearSolverMeta(type).isAvailable;
}

inline std::string_view preconditionerTypeName(PreconditionerType type) {
    switch (type) {
        case PreconditionerType::None:
            return "None";
        case PreconditionerType::Diagonal:
            return "Diagonal";
        case PreconditionerType::ICC:
            return "ICC";
        case PreconditionerType::ILU:
            return "ILU";
        case PreconditionerType::AdditiveSchwarz:
            return "AdditiveSchwarz";
        case PreconditionerType::AMG:
            return "AMG";
        case PreconditionerType::GaussSeidel:
            return "GaussSeidel";
        default:
            throw std::runtime_error("Unknown preconditioner type");
    }
}

inline std::vector<std::string> availableLinearSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::linearSolverRegistrySize; ++i) {
        if (detail::linearSolverRegistry[i].isAvailable) {
            result.emplace_back(detail::linearSolverRegistry[i].name);
        }
    }
    return result;
}

// =============================================================================
// Solver Configuration
// =============================================================================

struct SolverNodeConfig {
    SolverNodeRole role = SolverNodeRole::Preconditioner;
    PreconditionerType type = PreconditionerType::None;
    std::map<std::string, Real> parameters;
    std::vector<SolverNodeConfig> children;

    const SolverNodeConfig* findChild(SolverNodeRole childRole) const {
        for (const auto& child : children) {
            if (child.role == childRole) {
                return &child;
            }
        }
        return nullptr;
    }
};

struct SolverConfig {
    LinearSolverType linearType = LinearSolverType::Eigen_CG;
    int maxIterations = 1000;
    int restart = 30;
    Real relativeTolerance = 1e-10;
    Real absoluteTolerance = 1e-14;
    int printLevel = 0;
    Real preconditionerDropTolerance = 1e-4;
    int preconditionerFillLevel = 10;
    Real preconditionerShift = 1e-14;
    SolverNodeConfig preconditioner{SolverNodeRole::Preconditioner, PreconditionerType::Diagonal, {}, {}};

    const SolverNodeConfig& effectivePreconditioner() const {
        const SolverNodeConfig* current = &preconditioner;
        while (current) {
            if (current->type == PreconditionerType::AdditiveSchwarz) {
                if (const SolverNodeConfig* local = current->findChild(SolverNodeRole::LocalSolver)) {
                    current = local;
                    continue;
                }
            }
            if (current->type == PreconditionerType::AMG) {
                if (const SolverNodeConfig* smoother = current->findChild(SolverNodeRole::Smoother)) {
                    current = smoother;
                    continue;
                }
            }
            return *current;
        }
        return preconditioner;
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_CONFIG_HPP
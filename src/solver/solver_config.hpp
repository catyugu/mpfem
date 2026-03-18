#ifndef MPFEM_SOLVER_CONFIG_HPP
#define MPFEM_SOLVER_CONFIG_HPP

#include "core/types.hpp"
#include <string_view>
#include <stdexcept>
#include <vector>
#include <string>

namespace mpfem {

// =============================================================================
// Solver Type Enumeration
// =============================================================================

enum class SolverType {
    // Eigen solvers (always available)
    Eigen_SparseLU,       ///< Direct LU factorization
    Eigen_DGMRES_ILU,     ///< DGMRES with ILU preconditioner
    Eigen_MINRES,         ///< MINRES for symmetric indefinite systems

    // MKL PARDISO (conditionally available)
    MKL_Pardiso,          ///< MKL PARDISO direct solver
};

// =============================================================================
// Solver Metadata
// =============================================================================

struct SolverMeta {
    SolverType type;
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

inline constexpr SolverMeta solverRegistry[] = {
    // Eigen solvers (always available)
    {SolverType::Eigen_SparseLU,     "eigen.sparse_lu",     "Eigen SparseLU direct solver",         false, false, true},
    {SolverType::Eigen_DGMRES_ILU,   "eigen.dgmres_ilu",    "Eigen DGMRES with ILU preconditioner", true,  false, true},
    {SolverType::Eigen_MINRES,       "eigen.minres",        "Eigen MINRES (symmetric systems)",     true,  false, true},

    // MKL PARDISO
    {SolverType::MKL_Pardiso,        "mkl.pardiso",         "MKL PARDISO direct solver",            false, false, isMKLAvailable()},
};

inline constexpr size_t solverRegistrySize = sizeof(solverRegistry) / sizeof(SolverMeta);

}  // namespace detail

// =============================================================================
// Utility Functions
// =============================================================================

inline const SolverMeta& getSolverMeta(SolverType type) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].type == type) {
            return detail::solverRegistry[i];
        }
    }
    throw std::runtime_error("Unknown solver type");
}

inline const SolverMeta& getSolverMeta(std::string_view name) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].name == name) {
            return detail::solverRegistry[i];
        }
    }
    throw std::runtime_error("Unknown solver name: " + std::string(name));
}

inline std::string_view solverTypeName(SolverType type) {
    return getSolverMeta(type).name;
}

inline SolverType solverTypeFromName(std::string_view name) {
    return getSolverMeta(name).type;
}

inline bool isSolverAvailable(SolverType type) {
    return getSolverMeta(type).isAvailable;
}

inline bool isSolverAvailable(std::string_view name) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].name == name) {
            return detail::solverRegistry[i].isAvailable;
        }
    }
    return false;
}

inline std::vector<std::string> availableSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].isAvailable) {
            result.emplace_back(detail::solverRegistry[i].name);
        }
    }
    return result;
}

// =============================================================================
// Solver Configuration
// =============================================================================

struct SolverConfig {
    SolverType type = SolverType::Eigen_SparseLU;  // Default to Eigen SparseLU (always available)
    int maxIterations = 1000;
    int restart = 30;              // GMRES restart parameter
    Real relativeTolerance = 1e-10;
    Real absoluteTolerance = 1e-14;
    Real dropTolerance = 1e-4;     // ILU drop tolerance
    int fillFactor = 10;           // ILU fill factor
    int printLevel = 0;
    
    // Convenience constructors
    static SolverConfig eigenSparseLU() { return {SolverType::Eigen_SparseLU}; }
    static SolverConfig eigenDGMRES() { return {SolverType::Eigen_DGMRES_ILU}; }
    static SolverConfig eigenMINRES() { return {SolverType::Eigen_MINRES}; }
    static SolverConfig pardiso() { return {SolverType::MKL_Pardiso}; }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_CONFIG_HPP
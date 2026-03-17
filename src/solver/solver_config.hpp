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
    Eigen_CGIC,           ///< CG with Incomplete Cholesky (SPD only)
    Eigen_BiCGSTABILUT,   ///< BiCGSTAB with ILUT

    // External direct solvers (conditionally available)
    SuperLU_LU,           ///< SuperLU direct solver
    Umfpack_LU,           ///< SuiteSparse UMFPACK
    Cholmod_LLT,          ///< SuiteSparse CHOLMOD (SPD only)
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

inline constexpr bool isSuperLUAvailable() {
#ifdef MPFEM_USE_SUPERLU
    return true;
#else
    return false;
#endif
}

inline constexpr bool isUmfpackAvailable() {
#ifdef MPFEM_USE_UMFPACK
    return true;
#else
    return false;
#endif
}

inline constexpr bool isCholmodAvailable() {
#ifdef MPFEM_USE_CHOLMOD
    return true;
#else
    return false;
#endif
}

inline constexpr SolverMeta solverRegistry[] = {
    // Eigen solvers (always available)
    {SolverType::Eigen_SparseLU,     "eigen.sparse_lu",     "Eigen SparseLU direct solver",      false, false, true},
    {SolverType::Eigen_CGIC,         "eigen.cg_ic",         "Eigen CG with IC preconditioner",   true,  true,  true},
    {SolverType::Eigen_BiCGSTABILUT, "eigen.bicgstab_ilut", "Eigen BiCGSTAB with ILUT",          true,  false, true},

    // External solvers
    {SolverType::SuperLU_LU,   "superlu.lu",   "SuperLU direct solver",        false, false, isSuperLUAvailable()},
    {SolverType::Umfpack_LU,   "umfpack.lu",   "UMFPACK direct solver",        false, false, isUmfpackAvailable()},
    {SolverType::Cholmod_LLT,  "cholmod.llt",  "CHOLMOD Cholesky (SPD only)",  false, true,  isCholmodAvailable()},
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

inline std::vector<std::string> availableDirectSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].isAvailable && !detail::solverRegistry[i].isIterative) {
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
    Real relativeTolerance = 1e-10;
    Real absoluteTolerance = 1e-14;
    int printLevel = 0;
    
    // Convenience constructors
    static SolverConfig eigenSparseLU() { return {SolverType::Eigen_SparseLU}; }
    static SolverConfig superLU() { return {SolverType::SuperLU_LU}; }
    static SolverConfig umfpack() { return {SolverType::Umfpack_LU}; }
    static SolverConfig cholmod() { return {SolverType::Cholmod_LLT}; }
    static SolverConfig eigenCGIC() { return {SolverType::Eigen_CGIC}; }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_CONFIG_HPP
#ifndef MPFEM_SOLVER_CONFIG_HPP
#define MPFEM_SOLVER_CONFIG_HPP

#include "core/types.hpp"
#include <string_view>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

namespace mpfem {

// =============================================================================
// Solver Type Enumeration
// =============================================================================

enum class SolverType {
    // Eigen solvers (always available)
    Eigen_SparseLU,       ///< eigen.sparse_lu - Direct LU factorization
    Eigen_CG,             ///< eigen.cg - Conjugate Gradient (SPD only)
    Eigen_CGIC,           ///< eigen.cg_ic - CG with Incomplete Cholesky (SPD only)
    Eigen_BiCGSTAB,       ///< eigen.bicgstab - BiCGSTAB for non-symmetric
    Eigen_BiCGSTABILUT,   ///< eigen.bicgstab_ilut - BiCGSTAB with ILUT
    
    // External direct solvers (conditionally available)
    SuperLU_LU,           ///< superlu.lu - SuperLU direct solver
    Umfpack_LU,           ///< umfpack.lu - SuiteSparse UMFPACK
    
    // Special types
    Auto                  ///< Auto-select best available solver
};

// =============================================================================
// Solver Metadata (compile-time constants with availability)
// =============================================================================

struct SolverMeta {
    SolverType type;
    std::string_view name;    // Format: "package.algorithm"
    bool isIterative;         // true = iterative, false = direct
    bool requiresSPD;         // true = requires symmetric positive definite matrix
    bool isAvailable;         // compile-time availability check
    
    constexpr bool available() const { return isAvailable; }
};

// =============================================================================
// Solver Registry (Single unified table)
// =============================================================================

namespace detail {

// Compile-time availability checks
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

// Unified solver registry - single source of truth
inline constexpr SolverMeta solverRegistry[] = {
    // Eigen solvers (always available)
    {SolverType::Eigen_SparseLU,     "eigen.sparse_lu",      false, false, true},
    {SolverType::Eigen_CG,           "eigen.cg",             true,  true,  true},
    {SolverType::Eigen_CGIC,         "eigen.cg_ic",          true,  true,  true},
    {SolverType::Eigen_BiCGSTAB,     "eigen.bicgstab",       true,  false, true},
    {SolverType::Eigen_BiCGSTABILUT, "eigen.bicgstab_ilut",  true,  false, true},
    
    // External solvers
    {SolverType::SuperLU_LU,  "superlu.lu",    false, false, isSuperLUAvailable()},
    {SolverType::Umfpack_LU,  "umfpack.lu",    false, false, isUmfpackAvailable()},
};

inline constexpr size_t solverRegistrySize = sizeof(solverRegistry) / sizeof(SolverMeta);

}  // namespace detail

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get solver metadata by type.
 * @throws std::runtime_error if solver type not found
 */
inline const SolverMeta& getSolverMeta(SolverType type) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].type == type) {
            return detail::solverRegistry[i];
        }
    }
    throw std::runtime_error("Unknown solver type");
}

/**
 * @brief Get solver metadata by name.
 * @throws std::runtime_error if solver name not found
 */
inline const SolverMeta& getSolverMeta(std::string_view name) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].name == name) {
            return detail::solverRegistry[i];
        }
    }
    throw std::runtime_error("Unknown solver name: " + std::string(name));
}

/**
 * @brief Convert solver type to name string.
 */
inline std::string_view solverTypeName(SolverType type) {
    return getSolverMeta(type).name;
}

/**
 * @brief Convert name string to solver type.
 * @throws std::runtime_error if solver not found
 */
inline SolverType solverTypeFromName(std::string_view name) {
    return getSolverMeta(name).type;
}

/**
 * @brief Check if a solver type is available.
 */
inline bool isSolverAvailable(SolverType type) {
    return getSolverMeta(type).isAvailable;
}

/**
 * @brief Check if a solver is available by name.
 */
inline bool isSolverAvailable(std::string_view name) {
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].name == name) {
            return detail::solverRegistry[i].isAvailable;
        }
    }
    return false;
}

/**
 * @brief Get list of available solver names.
 */
inline std::vector<std::string> availableSolverNames() {
    std::vector<std::string> result;
    result.reserve(detail::solverRegistrySize);
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].isAvailable) {
            result.emplace_back(detail::solverRegistry[i].name);
        }
    }
    return result;
}

/**
 * @brief Get list of available direct solver names.
 */
inline std::vector<std::string> availableDirectSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].isAvailable && !detail::solverRegistry[i].isIterative) {
            result.emplace_back(detail::solverRegistry[i].name);
        }
    }
    return result;
}

/**
 * @brief Get list of available iterative solver names.
 */
inline std::vector<std::string> availableIterativeSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverRegistrySize; ++i) {
        if (detail::solverRegistry[i].isAvailable && detail::solverRegistry[i].isIterative) {
            result.emplace_back(detail::solverRegistry[i].name);
        }
    }
    return result;
}

// =============================================================================
// Solver Configuration Structure
// =============================================================================

/**
 * @brief Linear solver configuration.
 */
struct SolverConfig {
    SolverType type = SolverType::Auto;  // Use enum, not string
    int maxIterations = 1000;
    Real relativeTolerance = 1e-10;
    Real absoluteTolerance = 1e-14;
    int printLevel = 0;
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_CONFIG_HPP

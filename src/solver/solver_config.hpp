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
    MKL_Pardiso,          ///< mkl.pardiso - Intel MKL PARDISO
    Umfpack_LU,           ///< umfpack.lu - SuiteSparse UMFPACK
    
    // Special types
    Auto                  ///< Auto-select best available solver
};

// =============================================================================
// Solver Metadata (compile-time constants)
// =============================================================================

struct SolverMeta {
    SolverType type;
    std::string_view name;    // Format: "package.algorithm"
    bool isIterative;         // true = iterative, false = direct
    bool requiresSPD;         // true = requires symmetric positive definite matrix
};

// =============================================================================
// Availability Checks
// =============================================================================

namespace solver {

inline constexpr bool isSuperLUAvailable() {
#ifdef MPFEM_USE_SUPERLU
    return true;
#else
    return false;
#endif
}

inline constexpr bool isMKLAvailable() {
#ifdef MPFEM_USE_MKL
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

}  // namespace solver

// =============================================================================
// Solver Registry (Metadata only)
// =============================================================================

namespace detail {

// Solver metadata table - compile-time constants only
inline constexpr SolverMeta solverMetaTable[] = {
    // Eigen solvers (always available)
    {SolverType::Eigen_SparseLU,     "eigen.sparse_lu",      false, false},
    {SolverType::Eigen_CG,           "eigen.cg",             true,  true},
    {SolverType::Eigen_CGIC,         "eigen.cg_ic",          true,  true},
    {SolverType::Eigen_BiCGSTAB,     "eigen.bicgstab",       true,  false},
    {SolverType::Eigen_BiCGSTABILUT, "eigen.bicgstab_ilut",  true,  false},
    
    // External solvers
    {SolverType::SuperLU_LU,  "superlu.lu",    false, false},
    {SolverType::MKL_Pardiso, "mkl.pardiso",   false, false},
    {SolverType::Umfpack_LU,  "umfpack.lu",    false, false},
};

inline constexpr size_t solverMetaSize = sizeof(solverMetaTable) / sizeof(SolverMeta);

// Availability table
inline constexpr bool solverAvailability[] = {
    // Eigen solvers
    true,   // Eigen_SparseLU
    true,   // Eigen_CG
    true,   // Eigen_CGIC
    true,   // Eigen_BiCGSTAB
    true,   // Eigen_BiCGSTABILUT
    // External solvers
    solver::isSuperLUAvailable(),   // SuperLU_LU
    solver::isMKLAvailable(),       // MKL_Pardiso
    solver::isUmfpackAvailable(),   // Umfpack_LU
};

}  // namespace detail

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get solver metadata by type.
 * @throws std::runtime_error if solver type not found
 */
inline const SolverMeta& getSolverMeta(SolverType type) {
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverMetaTable[i].type == type) {
            return detail::solverMetaTable[i];
        }
    }
    throw std::runtime_error("Unknown solver type");
}

/**
 * @brief Get solver metadata by name.
 * @throws std::runtime_error if solver name not found
 */
inline const SolverMeta& getSolverMeta(std::string_view name) {
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverMetaTable[i].name == name) {
            return detail::solverMetaTable[i];
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
    const auto& meta = getSolverMeta(type);
    const size_t idx = static_cast<size_t>(&meta - detail::solverMetaTable);
    if (idx < detail::solverMetaSize) {
        return detail::solverAvailability[idx];
    }
    return false;
}

/**
 * @brief Check if a solver is available by name.
 */
inline bool isSolverAvailable(std::string_view name) {
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverMetaTable[i].name == name) {
            return detail::solverAvailability[i];
        }
    }
    return false;
}

/**
 * @brief Get list of available solver names.
 */
inline std::vector<std::string> availableSolverNames() {
    std::vector<std::string> result;
    result.reserve(detail::solverMetaSize);
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverAvailability[i]) {
            result.emplace_back(detail::solverMetaTable[i].name);
        }
    }
    return result;
}

/**
 * @brief Get list of available direct solver names.
 */
inline std::vector<std::string> availableDirectSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverAvailability[i] && !detail::solverMetaTable[i].isIterative) {
            result.emplace_back(detail::solverMetaTable[i].name);
        }
    }
    return result;
}

/**
 * @brief Get list of available iterative solver names.
 */
inline std::vector<std::string> availableIterativeSolverNames() {
    std::vector<std::string> result;
    for (size_t i = 0; i < detail::solverMetaSize; ++i) {
        if (detail::solverAvailability[i] && detail::solverMetaTable[i].isIterative) {
            result.emplace_back(detail::solverMetaTable[i].name);
        }
    }
    return result;
}

}  // namespace mpfem

#endif  // MPFEM_SOLVER_CONFIG_HPP
#ifndef MPFEM_SOLVER_CONFIG_HPP
#define MPFEM_SOLVER_CONFIG_HPP

#include "core/types.hpp"
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mpfem {

    // =============================================================================
    // Forward Declarations
    // =============================================================================

    class Preconditioner;

    // =============================================================================
    // Solver Type Enumerations
    // =============================================================================

    enum class LinearSolverType {
        // Eigen solvers (always available)
        Eigen_SparseLU, ///< Direct LU factorization
        Eigen_CG, ///< Conjugate Gradient
        Eigen_DGMRES, ///< DGMRES style Krylov solver

        // MKL PARDISO (conditionally available)
        MKL_Pardiso, ///< MKL PARDISO direct solver

        // SuiteSparse UMFPACK (conditionally available)
        UMFPACK_LU, ///< SuiteSparse UMFPACK direct solver
    };

    enum class PreconditionerType {
        None,
        Diagonal,
        ICC,
        ILU,
        AdditiveSchwarz
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

        inline constexpr bool isMKLAvailable()
        {
#ifdef MPFEM_USE_MKL
            return true;
#else
            return false;
#endif
        }

        inline constexpr bool isSuiteSparseAvailable()
        {
#ifdef MPFEM_USE_SUITESPARSE
            return true;
#else
            return false;
#endif
        }

        inline constexpr LinearSolverMeta linearSolverRegistry[] = {
            // Eigen solvers (always available)
            {LinearSolverType::Eigen_SparseLU, "SparseLU", "Eigen SparseLU direct solver", false, false, true},
            {LinearSolverType::Eigen_CG, "CG", "Eigen Conjugate Gradient solver", true, true, true},
            {LinearSolverType::Eigen_DGMRES, "DGMRES", "Eigen DGMRES solver", true, false, true},

            // MKL PARDISO
            {LinearSolverType::MKL_Pardiso, "Pardiso", "MKL PARDISO direct solver", false, false, isMKLAvailable()},

            // SuiteSparse UMFPACK
            {LinearSolverType::UMFPACK_LU, "UMFPACK", "SuiteSparse UMFPACK direct solver", false, false, isSuiteSparseAvailable()},
        };

        inline constexpr size_t linearSolverRegistrySize = sizeof(linearSolverRegistry) / sizeof(LinearSolverMeta);

    } // namespace detail

    // =============================================================================
    // Utility Functions
    // =============================================================================

    inline const LinearSolverMeta& getLinearSolverMeta(LinearSolverType type)
    {
        for (size_t i = 0; i < detail::linearSolverRegistrySize; ++i) {
            if (detail::linearSolverRegistry[i].type == type) {
                return detail::linearSolverRegistry[i];
            }
        }
        throw std::runtime_error("Unknown linear solver type");
    }

    inline std::string_view linearSolverTypeName(LinearSolverType type)
    {
        return getLinearSolverMeta(type).name;
    }

    inline bool isLinearSolverAvailable(LinearSolverType type)
    {
        return getLinearSolverMeta(type).isAvailable;
    }

    inline std::string_view preconditionerTypeName(PreconditionerType type)
    {
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
        default:
            throw std::runtime_error("Unknown preconditioner type");
        }
    }

    inline std::vector<std::string> availableLinearSolverNames()
    {
        std::vector<std::string> result;
        for (size_t i = 0; i < detail::linearSolverRegistrySize; ++i) {
            if (detail::linearSolverRegistry[i].isAvailable) {
                result.emplace_back(detail::linearSolverRegistry[i].name);
            }
        }
        return result;
    }

    // =============================================================================
    // Linear Solver Configuration
    // =============================================================================

    struct LinearSolverConfig {
        LinearSolverType type = LinearSolverType::Eigen_CG;
        int maxIterations = 1000;
        int restart = 30;
        Real tolerance = 1e-10;
        int printLevel = 0;
    };

    // =============================================================================
    // Preconditioner Configuration (hierarchical)
    // =============================================================================

    struct PreconditionerConfig {
        PreconditionerType type = PreconditionerType::None;
        std::map<std::string, Real> parameters;
        PreconditionerConfig* localSolver = nullptr; ///< For AdditiveSchwarz (non-owning)
        PreconditionerConfig* coarseSolver = nullptr; ///< For AdditiveSchwarz/AMG (non-owning)
        PreconditionerConfig* smoother = nullptr; ///< For AMG (non-owning)
    };

    // =============================================================================
    // Solver Configuration
    // =============================================================================

    struct SolverConfig {
        LinearSolverConfig solver;
        PreconditionerConfig preconditioner;
    };

} // namespace mpfem

#endif // MPFEM_SOLVER_CONFIG_HPP
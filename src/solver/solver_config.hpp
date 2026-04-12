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

    class LinearOperator;

    // =============================================================================
    // Operator Type Enumeration (unified for solvers and preconditioners)
    // =============================================================================

    enum class OperatorType {
        // Direct solvers
        SparseLU,
        Pardiso,
        Umfpack,

        // Iterative solvers
        CG,
        DGMRES,

        // Preconditioners
        Diagonal,
        ICC,
        ILU,
        AdditiveSchwarz,
    };

    // =============================================================================
    // Operator Metadata
    // =============================================================================

    struct OperatorMeta {
        OperatorType type;
        std::string_view name;
        std::string_view description;
        bool isIterative;
        bool requiresSPD;
        bool isAvailable;
    };

    // =============================================================================
    // Operator Registry
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

        inline constexpr OperatorMeta operatorRegistry[] = {
            // Direct solvers
            {OperatorType::SparseLU, "SparseLU", "Eigen SparseLU direct solver", false, false, true},
            {OperatorType::Pardiso, "Pardiso", "MKL PARDISO direct solver", false, false, isMKLAvailable()},
            {OperatorType::Umfpack, "UMFPACK", "SuiteSparse UMFPACK direct solver", false, false, isSuiteSparseAvailable()},

            // Iterative solvers
            {OperatorType::CG, "CG", "Eigen Conjugate Gradient solver", true, true, true},
            {OperatorType::DGMRES, "DGMRES", "Eigen DGMRES solver", true, false, true},

            // Preconditioners
            {OperatorType::Diagonal, "Diagonal", "Diagonal (Jacobi) preconditioner", false, false, true},
            {OperatorType::ICC, "ICC", "Incomplete Cholesky preconditioner", false, true, true},
            {OperatorType::ILU, "ILU", "Incomplete LU preconditioner", false, false, true},
            {OperatorType::AdditiveSchwarz, "AdditiveSchwarz", "Additive Schwarz domain decomposition", false, false, true},
        };

        inline constexpr size_t operatorRegistrySize = sizeof(operatorRegistry) / sizeof(OperatorMeta);

    } // namespace detail

    // =============================================================================
    // Utility Functions
    // =============================================================================

    inline const OperatorMeta& getOperatorMeta(OperatorType type)
    {
        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (detail::operatorRegistry[i].type == type) {
                return detail::operatorRegistry[i];
            }
        }
        throw std::runtime_error("Unknown operator type");
    }

    inline std::string_view operatorTypeName(OperatorType type)
    {
        return getOperatorMeta(type).name;
    }

    inline bool isOperatorAvailable(OperatorType type)
    {
        return getOperatorMeta(type).isAvailable;
    }

    inline std::vector<std::string> availableOperatorNames()
    {
        std::vector<std::string> result;
        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (detail::operatorRegistry[i].isAvailable) {
                result.emplace_back(detail::operatorRegistry[i].name);
            }
        }
        return result;
    }

    inline OperatorType operatorTypeFromName(std::string_view name)
    {
        // Case-insensitive comparison helper
        auto iequals = [](std::string_view a, std::string_view b) {
            if (a.size() != b.size())
                return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
                    return false;
            }
            return true;
        };

        // Normalize input: lowercase, alphanumeric only
        std::string normalized;
        normalized.reserve(name.size());
        for (char ch : name) {
            const unsigned char value = static_cast<unsigned char>(ch);
            if (std::isalnum(value)) {
                normalized.push_back(static_cast<char>(std::tolower(value)));
            }
        }

        // Search registry (single source of truth)
        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (iequals(detail::operatorRegistry[i].name, normalized)) {
                return detail::operatorRegistry[i].type;
            }
        }

        throw std::runtime_error("Unknown operator type: " + std::string(name));
    }

    // =============================================================================
    // Recursive Operator Configuration (onion skin structure)
    // =============================================================================

    /**
     * @brief Recursive configuration for LinearOperator tree.
     *
     * Supports arbitrary nesting of preconditioners via the preconditioner field.
     * Example XML structure:
     * <Operator type="CG">
     *   <Tolerance>1e-10</Tolerance>
     *   <MaxIterations>1000</MaxIterations>
     *   <Preconditioner type="AdditiveSchwarz">
     *     <Overlap>1</Overlap>
     *     <LocalSolver>
     *       <Preconditioner type="ILU"/>
     *     </LocalSolver>
     *   </Preconditioner>
     * </Operator>
     */
    struct LinearOperatorConfig {
        /// Operator type
        OperatorType type = OperatorType::CG;

        /// Scalar parameters (Tolerance, MaxIterations, Sweeps, etc.)
        std::map<std::string, Real> parameters;

        /// Nested preconditioner (onion skin)
        std::unique_ptr<LinearOperatorConfig> preconditioner;

        /// For AdditiveSchwarz: local solver configuration
        std::unique_ptr<LinearOperatorConfig> localSolver;

        /// For AdditiveSchwarz/AMG: coarse solver configuration
        std::unique_ptr<LinearOperatorConfig> coarseSolver;

        /// For AMG: smoother configuration
        std::unique_ptr<LinearOperatorConfig> smoother;

        // Convenience constructor
        LinearOperatorConfig() = default;
        explicit LinearOperatorConfig(OperatorType t) : type(t) { }
    };

} // namespace mpfem

#endif // MPFEM_SOLVER_CONFIG_HPP
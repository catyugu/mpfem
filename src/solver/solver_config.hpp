#ifndef MPFEM_SOLVER_CONFIG_HPP
#define MPFEM_SOLVER_CONFIG_HPP

#include "core/types.hpp"
#include <map>
#include <memory>
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
    // Utility Functions (declarations)
    // =============================================================================

    const OperatorMeta& getOperatorMeta(OperatorType type);
    std::string_view operatorTypeName(OperatorType type);
    bool isOperatorAvailable(OperatorType type);
    std::vector<std::string> availableOperatorNames();
    OperatorType operatorTypeFromName(std::string_view name);

    // =============================================================================
    // Recursive Operator Configuration (onion skin structure)
    // =============================================================================

    /**
     * @brief Recursive configuration for LinearOperator tree.
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
        explicit LinearOperatorConfig(OperatorType t)
            : type(t) { }
        
        // Deep copy constructor needed because of unique_ptr
        LinearOperatorConfig(const LinearOperatorConfig& other);
        LinearOperatorConfig& operator=(const LinearOperatorConfig& other);

        LinearOperatorConfig(LinearOperatorConfig&&) = default;
        LinearOperatorConfig& operator=(LinearOperatorConfig&&) = default;
    };

} // namespace mpfem

#endif // MPFEM_SOLVER_CONFIG_HPP

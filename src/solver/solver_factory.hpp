#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "linear_operator.hpp"
#include "solver_config.hpp"
#include <memory>

namespace mpfem {

    // =============================================================================
    // Operator Factory
    // =============================================================================

    /**
     * @brief Factory for creating LinearOperator instances from configuration.
     *
     * Recursively parses LinearOperatorConfig tree and instantiates operators,
     * wiring nested preconditioners via set_preconditioner().
     *
     * Design principles:
     * - No fallback logic - if operator is not available, throw exception
     * - Recursive construction for nested preconditioners
     * - All operators inherit from LinearOperator base class
     */
    class OperatorFactory {
    public:
        /**
         * @brief Create a LinearOperator from configuration.
         */
        static std::unique_ptr<LinearOperator> create(const LinearOperatorConfig& config);

        /// Create operator by type
        static std::unique_ptr<LinearOperator> createByType(OperatorType type);
    };

} // namespace mpfem

#endif // MPFEM_SOLVER_FACTORY_HPP
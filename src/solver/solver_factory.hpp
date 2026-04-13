#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "core/logger.hpp"
#include "eigen_solver.hpp"
#include "linear_operator.hpp"
#include "pardiso_solver.hpp"
#include "solver_config.hpp"
#include "umfpack_solver.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

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
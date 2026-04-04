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
        static std::unique_ptr<LinearOperator> create(const LinearOperatorConfig& config)
        {
            const auto& meta = getOperatorMeta(config.type);
            if (!meta.isAvailable) {
                throw std::runtime_error(
                    "Operator '" + std::string(meta.name) + "' is not available.");
            }

            std::unique_ptr<LinearOperator> op = createByType(config.type);

            // Apply parameters
            applyParameters(op.get(), config);

            // Create and attach nested preconditioner
            if (config.preconditioner) {
                auto pc = create(*config.preconditioner);
                op->set_preconditioner(std::move(pc));
            }

            return op;
        }

        /// Create operator by type
        static std::unique_ptr<LinearOperator> createByType(OperatorType type)
        {
            switch (type) {
            case OperatorType::SparseLU:
                return std::make_unique<EigenSparseLUOperator>();
            case OperatorType::CG:
                return std::make_unique<CgOperator>();
            case OperatorType::DGMRES:
                return std::make_unique<GmresOperator>();
            case OperatorType::Diagonal:
                return std::make_unique<DiagonalOperator>();
            case OperatorType::ICC:
                return std::make_unique<IccOperator>();
            case OperatorType::ILU:
                return std::make_unique<IluOperator>();
            case OperatorType::AdditiveSchwarz:
                return std::make_unique<AdditiveSchwarzOperator>();
            case OperatorType::Pardiso:
#ifdef MPFEM_USE_MKL
                return std::make_unique<PardisoSolver>();
#else
                throw std::runtime_error("PardisoOperator: MKL not available");
#endif
            case OperatorType::Umfpack:
#ifdef MPFEM_USE_SUITESPARSE
                return std::make_unique<UmfpackSolver>();
#else
                throw std::runtime_error("UmfpackOperator: SuiteSparse not available");
#endif
            default:
                throw std::runtime_error("Unsupported operator type");
            }
        }

    private:
        static void applyParameters(LinearOperator* op, const LinearOperatorConfig& config)
        {
            const auto& params = config.parameters;

            if (auto* cg = dynamic_cast<CgOperator*>(op)) {
                if (auto it = params.find("MaxIterations"); it != params.end()) {
                    cg->set_max_iterations(static_cast<int>(it->second));
                }
                if (auto it = params.find("Tolerance"); it != params.end()) {
                    cg->set_tolerance(it->second);
                }
            }
            else if (auto* gmres = dynamic_cast<GmresOperator*>(op)) {
                if (auto it = params.find("MaxIterations"); it != params.end()) {
                    gmres->set_max_iterations(static_cast<int>(it->second));
                }
                if (auto it = params.find("Tolerance"); it != params.end()) {
                    gmres->set_tolerance(it->second);
                }
            }
            else if (auto* icc = dynamic_cast<IccOperator*>(op)) {
                if (auto it = params.find("Shift"); it != params.end()) {
                    icc->set_shift(it->second);
                }
            }
        }
    };

    // Backward compatibility alias
    using SolverFactory = OperatorFactory;

} // namespace mpfem

#endif // MPFEM_SOLVER_FACTORY_HPP
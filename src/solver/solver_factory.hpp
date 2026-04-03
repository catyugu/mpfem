#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "core/logger.hpp"
#include "eigen_solver.hpp"
#include "linear_solver.hpp"
#include "pardiso_solver.hpp"
#include "preconditioner.hpp"
#include "solver_config.hpp"
#include "umfpack_solver.hpp"
#include <cctype>
#include <memory>
#include <string_view>

namespace mpfem {

    // =============================================================================
    // Solver Factory
    // =============================================================================

    /**
     * @brief Factory for creating linear solvers and preconditioners.
     *
     * Design principles:
     * - No fallback logic - if solver is not available, throw exception
     * - Separate creation for solver and preconditioner
     * - Direct solvers (SparseLU, Pardiso, UMFPACK) do not support preconditioners
     * - setPreconditioner() will be called on the solver to attach the preconditioner
     */
    class SolverFactory {
    public:
        /**
         * @brief Convert string to LinearSolverType enum.
         * @param name Solver name (e.g., "CG", "DGMRES", "SparseLU", "Pardiso", "UMFPACK")
         * @throws std::runtime_error if name is not recognized
         */
        static LinearSolverType linearSolverTypeFromName(std::string_view name)
        {
            const std::string key = normalizeToken(name);
            if (key == "cg") {
                return LinearSolverType::Eigen_CG;
            }
            if (key == "dgmres") {
                return LinearSolverType::Eigen_DGMRES;
            }
            if (key == "sparselu") {
                return LinearSolverType::Eigen_SparseLU;
            }
            if (key == "pardiso") {
                return LinearSolverType::MKL_Pardiso;
            }
            if (key == "umfpack") {
                return LinearSolverType::UMFPACK_LU;
            }

            throw std::runtime_error("Unsupported LinearSolver type: " + std::string(name));
        }

        /**
         * @brief Convert string to PreconditionerType enum.
         * @param name Preconditioner name (e.g., "None", "Diagonal", "ICC", "ILU")
         * @throws std::runtime_error if name is not recognized
         */
        static PreconditionerType preconditionerTypeFromName(std::string_view name)
        {
            const std::string key = normalizeToken(name);
            if (key == "none") {
                return PreconditionerType::None;
            }
            if (key == "diagonal") {
                return PreconditionerType::Diagonal;
            }
            if (key == "icc") {
                return PreconditionerType::ICC;
            }
            if (key == "ilu") {
                return PreconditionerType::ILU;
            }
            if (key == "additiveschwarz") {
                return PreconditionerType::AdditiveSchwarz;
            }

            throw std::runtime_error("Unsupported Preconditioner type: " + std::string(name));
        }

        /**
         * @brief Create a solver instance without preconditioner.
         * @param config Solver configuration containing linearType
         * @return Unique pointer to the solver instance
         * @throws std::runtime_error if solver is not available
         */
        static std::unique_ptr<LinearSolver> createSolver(const SolverConfig& config)
        {
            const LinearSolverType type = config.solver.type;

            // Check availability
            const auto& meta = getLinearSolverMeta(type);
            if (!meta.isAvailable) {
                throw std::runtime_error(
                    "Solver '" + std::string(meta.name) + "' is not available. "
                                                          "Available solvers: "
                    + joinSolverNames());
            }

            LOG_DEBUG << "Creating solver: " << meta.name;

            // Create solver instance
            auto solver = createSolverByType(type);

            // Apply configuration
            solver->setMaxIterations(config.solver.maxIterations);
            solver->setTolerance(config.solver.tolerance);
            solver->setPrintLevel(config.solver.printLevel);

            // Apply solver-specific configuration
            solver->applyConfig(config);

            return solver;
        }

        /**
         * @brief Create a preconditioner instance.
         * @param config Preconditioner configuration
         * @return Unique pointer to the preconditioner, or nullptr if not implemented
         *         (AdditiveSchwarz) or type is None
         */
        static std::unique_ptr<Preconditioner> createPreconditioner(const PreconditionerConfig& config)
        {
            const PreconditionerType type = config.type;

            // AdditiveSchwarz require hierarchical setup - not implemented here
            if (type == PreconditionerType::AdditiveSchwarz) {
                LOG_WARN << "AdditiveSchwarz preconditioner requires hierarchical setup; "
                         << "returning nullptr. Use HierarchicalSolver instead.";
                return nullptr;
            }

            // Use registry for simple types (None, Diagonal, ICC, ILU)
            auto precond = PreconditionerRegistry::create(type);
            if (precond && !config.parameters.empty()) {
                precond->setParameters(config.parameters);
            }
            return precond;
        }

        /**
         * @brief Create a composed solver with optional preconditioner.
         * @param config Full solver configuration with solver type and preconditioner
         * @return Unique pointer to the solver with preconditioner attached
         * @throws std::runtime_error if solver is not available or preconditioner
         *         is not supported (direct solvers don't support preconditioners)
         */
        static std::unique_ptr<LinearSolver> create(const SolverConfig& config)
        {
            // Create solver
            auto solver = createSolver(config);

            // Create preconditioner if requested
            if (config.preconditioner.type != PreconditionerType::None) {
                auto preconditioner = createPreconditioner(config.preconditioner);
                if (preconditioner) {
                    solver->setPreconditioner(std::move(preconditioner));
                }
            }

            return solver;
        }

    private:
        static std::string normalizeToken(std::string_view text)
        {
            std::string normalized;
            normalized.reserve(text.size());
            for (char ch : text) {
                const unsigned char value = static_cast<unsigned char>(ch);
                if (std::isalnum(value)) {
                    normalized.push_back(static_cast<char>(std::tolower(value)));
                }
            }
            return normalized;
        }

        static std::unique_ptr<LinearSolver> createSolverByType(LinearSolverType type)
        {
            switch (type) {
            // Eigen solvers (always available)
            case LinearSolverType::Eigen_SparseLU:
                return std::make_unique<EigenSparseLUSolver>();
            case LinearSolverType::Eigen_CG:
                return std::make_unique<EigenCGSolver>();
            case LinearSolverType::Eigen_DGMRES:
                return std::make_unique<EigenDGMRESSolver>();

            // MKL PARDISO
            case LinearSolverType::MKL_Pardiso:
                return std::make_unique<PardisoSolver>();

            // SuiteSparse UMFPACK
            case LinearSolverType::UMFPACK_LU:
                return std::make_unique<UmfpackSolver>();

            default:
                throw std::runtime_error("Unsupported solver type");
            }
        }

        static std::string joinSolverNames()
        {
            auto names = availableLinearSolverNames();
            std::string result;
            for (size_t i = 0; i < names.size(); ++i) {
                if (i > 0)
                    result += ", ";
                result += names[i];
            }
            return result;
        }
    };

} // namespace mpfem

#endif // MPFEM_SOLVER_FACTORY_HPP
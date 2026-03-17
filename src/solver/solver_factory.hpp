#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "solver_config.hpp"
#include "linear_solver.hpp"
#include "eigen_solver.hpp"
#include "superlu_solver.hpp"
#include "umfpack_solver.hpp"
#include "core/logger.hpp"
#include <memory>

namespace mpfem {

// =============================================================================
// Solver Factory
// =============================================================================

/**
 * @brief Factory for creating linear solvers.
 * 
 * Design principles:
 * - No fallback logic - if solver is not available, throw exception
 * - Single entry point: create(const SolverConfig&)
 * - Configuration must specify an explicit solver type
 */
class SolverFactory {
public:
    /**
     * @brief Create a solver from configuration.
     * @param config Solver configuration with explicit type
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(const SolverConfig& config) {
        const SolverType type = config.type;
        
        // Check availability
        const auto& meta = getSolverMeta(type);
        if (!meta.isAvailable) {
            throw std::runtime_error(
                "Solver '" + std::string(meta.name) + "' is not available. "
                "Available solvers: " + joinSolverNames());
        }
        
        LOG_DEBUG << "Creating solver: " << meta.name;
        
        // Create solver instance
        auto solver = createByType(type);
        
        // Apply configuration
        solver->setMaxIterations(config.maxIterations);
        solver->setTolerance(config.relativeTolerance);
        solver->setPrintLevel(config.printLevel);
        
        return solver;
    }
    
    /**
     * @brief Create a solver by type.
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(SolverType type) {
        const auto& meta = getSolverMeta(type);
        if (!meta.isAvailable) {
            throw std::runtime_error(
                "Solver '" + std::string(meta.name) + "' is not available. "
                "Available solvers: " + joinSolverNames());
        }
        return createByType(type);
    }
    
    /**
     * @brief Create a solver by name.
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(std::string_view name) {
        return create(solverTypeFromName(name));
    }
    
    /**
     * @brief Get list of all available solver names.
     */
    static std::vector<std::string> availableSolvers() {
        return availableSolverNames();
    }
    
    /**
     * @brief Check if a solver is available.
     */
    static bool isAvailable(SolverType type) {
        return isSolverAvailable(type);
    }
    
    static bool isAvailable(std::string_view name) {
        return isSolverAvailable(name);
    }

private:
    static std::unique_ptr<LinearSolver> createByType(SolverType type) {
        switch (type) {
            // Eigen solvers (always available)
            case SolverType::Eigen_SparseLU:
                return std::make_unique<EigenSparseLUSolver>();
            case SolverType::Eigen_CGIC:
                return std::make_unique<EigenCGICSolver>();
            case SolverType::Eigen_BiCGSTABILUT:
                return std::make_unique<EigenBiCGSTABILUTSolver>();

            // External solvers
            case SolverType::SuperLU_LU:
                return std::make_unique<SuperLUSolver>();
            case SolverType::Umfpack_LU:
                return std::make_unique<UmfpackSolver>();

            default:
                throw std::runtime_error("Unsupported solver type");
        }
    }
    
    static std::string joinSolverNames() {
        auto names = availableSolverNames();
        std::string result;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i > 0) result += ", ";
            result += names[i];
        }
        return result;
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_FACTORY_HPP

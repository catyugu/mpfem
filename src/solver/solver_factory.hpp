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
 * Single entry point: create(const SolverConfig&)
 * No fallback logic - throws exception if solver is not available.
 */
class SolverFactory {
public:
    /**
     * @brief Create a solver from configuration.
     * @param config Solver configuration
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(const SolverConfig& config) {
        SolverType type = config.type;
        
        // Handle Auto type
        if (type == SolverType::Auto) {
            type = selectBestDirectSolver();
        }
        
        // Check availability
        const auto& meta = getSolverMeta(type);
        if (!meta.isAvailable) {
            throw std::runtime_error("Solver '" + std::string(meta.name) + 
                "' is not available. Please rebuild with the required dependency.");
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
     * @brief Get list of all available solvers.
     */
    static std::vector<std::string> availableSolvers() {
        return availableSolverNames();
    }

private:
    /**
     * @brief Create solver by type (no configuration applied).
     */
    static std::unique_ptr<LinearSolver> createByType(SolverType type) {
        switch (type) {
            // Eigen solvers (always available)
            case SolverType::Eigen_SparseLU:
                return std::make_unique<EigenSparseLUSolver>();
            case SolverType::Eigen_CG:
                return std::make_unique<EigenCGSolver>();
            case SolverType::Eigen_CGIC:
                return std::make_unique<EigenCGICSolver>();
            case SolverType::Eigen_BiCGSTAB:
                return std::make_unique<EigenBiCGSTABSolver>();
            case SolverType::Eigen_BiCGSTABILUT:
                return std::make_unique<EigenBiCGSTABILUTSolver>();
            
            // External solvers (conditionally available)
            case SolverType::SuperLU_LU:
                return std::make_unique<SuperLUSolver>();
            case SolverType::Umfpack_LU:
                return std::make_unique<UmfpackSolver>();
            
            default:
                throw std::runtime_error("Unknown solver type");
        }
    }
    
    /**
     * @brief Select best available direct solver for Auto type.
     * Priority: SuperLU > UMFPACK > Eigen SparseLU
     */
    static SolverType selectBestDirectSolver() {
#ifdef MPFEM_USE_SUPERLU
        LOG_DEBUG << "Auto-selecting SuperLU solver";
        return SolverType::SuperLU_LU;
#elif defined(MPFEM_USE_UMFPACK)
        LOG_DEBUG << "Auto-selecting UMFPACK solver";
        return SolverType::Umfpack_LU;
#else
        LOG_DEBUG << "Auto-selecting Eigen SparseLU solver";
        return SolverType::Eigen_SparseLU;
#endif
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_FACTORY_HPP
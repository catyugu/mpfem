#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "solver_config.hpp"
#include "linear_solver.hpp"
#include "eigen_solver.hpp"
#include "superlu_solver.hpp"
#include "umfpack_solver.hpp"
#include "core/logger.hpp"
#include <memory>
#include <string>

namespace mpfem {

// =============================================================================
// Solver Factory
// =============================================================================

/**
 * @brief Factory for creating linear solvers.
 * 
 * Creates solver instances based on type string or enum.
 * No fallback logic - throws exception if solver is not available.
 * 
 * Adding a new solver:
 * 1. Create the solver class (e.g., MySolver)
 * 2. Add entry to SolverType enum in solver_config.hpp
 * 3. Add metadata entry to solverMetaTable in solver_config.hpp
 * 4. Add availability entry to solverAvailability in solver_config.hpp
 * 5. Add case in create() method below
 */
class SolverFactory {
public:
    /**
     * @brief Create a solver by type enum.
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(SolverType type) {
        if (type == SolverType::Auto) {
            return createAuto();
        }
        
        if (!isSolverAvailable(type)) {
            const auto& meta = getSolverMeta(type);
            throw std::runtime_error("Solver '" + std::string(meta.name) + 
                "' is not available. Please rebuild with the required dependency.");
        }
        
        LOG_DEBUG << "Creating solver: " << solverTypeName(type);
        
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
     * @brief Create a solver by name string.
     * @throws std::runtime_error if solver is not found or not available
     */
    static std::unique_ptr<LinearSolver> create(const std::string& name) {
        return create(solverTypeFromName(name));
    }
    
    /**
     * @brief Create a solver with configuration.
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(
        SolverType type,
        int maxIterations,
        Real tolerance,
        int printLevel = 0)
    {
        auto solver = create(type);
        solver->setMaxIterations(maxIterations);
        solver->setTolerance(tolerance);
        solver->setPrintLevel(printLevel);
        return solver;
    }
    
    /**
     * @brief Create a solver with configuration from string.
     * @throws std::runtime_error if solver is not found or not available
     */
    static std::unique_ptr<LinearSolver> create(
        const std::string& name,
        int maxIterations,
        Real tolerance,
        int printLevel = 0)
    {
        return create(solverTypeFromName(name), maxIterations, tolerance, printLevel);
    }
    
    /**
     * @brief Auto-select the best available direct solver.
     * 
     * Priority:
     * 1. SuperLU (if available)
     * 2. UMFPACK (if available)
     * 3. Eigen SparseLU (always available)
     */
    static std::unique_ptr<LinearSolver> createAuto() {
#ifdef MPFEM_USE_SUPERLU
        LOG_DEBUG << "Auto-selecting SuperLU solver";
        return std::make_unique<SuperLUSolver>();
#elif defined(MPFEM_USE_UMFPACK)
        LOG_DEBUG << "Auto-selecting UMFPACK solver";
        return std::make_unique<UmfpackSolver>();
#else
        LOG_DEBUG << "Auto-selecting Eigen SparseLU solver";
        return std::make_unique<EigenSparseLUSolver>();
#endif
    }
    
    /**
     * @brief Get a solver suitable for symmetric positive definite matrices.
     * 
     * For SPD matrices, CG with Incomplete Cholesky is usually best.
     */
    static std::unique_ptr<LinearSolver> createForSPD(
        int maxIterations = 1000,
        Real tolerance = 1e-10,
        int printLevel = 0)
    {
        auto solver = std::make_unique<EigenCGICSolver>();
        solver->setMaxIterations(maxIterations);
        solver->setTolerance(tolerance);
        solver->setPrintLevel(printLevel);
        return solver;
    }
    
    /**
     * @brief Get a solver suitable for general non-symmetric matrices.
     * 
     * Uses auto-selection for direct solvers.
     */
    static std::unique_ptr<LinearSolver> createForGeneral(
        int maxIterations = 1000,
        Real tolerance = 1e-10,
        int printLevel = 0)
    {
        auto solver = createAuto();
        solver->setMaxIterations(maxIterations);
        solver->setTolerance(tolerance);
        solver->setPrintLevel(printLevel);
        return solver;
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_FACTORY_HPP

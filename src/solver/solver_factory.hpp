#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "linear_solver.hpp"
#include "eigen_solver.hpp"
#include "pardiso_solver.hpp"
#include "core/logger.hpp"
#include <memory>
#include <string>

namespace mpfem {

/**
 * @brief Factory for creating linear solvers.
 * 
 * Creates solver instances based on type string or enum.
 * Supports runtime selection and configuration.
 */
class SolverFactory {
public:
    /**
     * @brief Create a solver by type enum.
     */
    static std::unique_ptr<LinearSolver> create(SolverType type) {
        switch (type) {
            case SolverType::SparseLU:
                return std::make_unique<EigenSparseLUSolver>();
                
            case SolverType::SparseQR:
                return std::make_unique<EigenSparseQRSolver>();
                
            case SolverType::Pardiso:
                return std::make_unique<PardisoSolver>();
                
            case SolverType::CG:
                return std::make_unique<EigenCGSolver>();
                
            case SolverType::CGWithIC:
                return std::make_unique<EigenCGICSolver>();
                
            case SolverType::BiCGSTAB:
                return std::make_unique<EigenBiCGSTABSolver>();
                
            case SolverType::BiCGSTABWithILUT:
                return std::make_unique<EigenBiCGSTABILUTSolver>();
                
            case SolverType::Auto:
                return createAuto();
                
            default:
                LOG_ERROR << "Unknown solver type, using SparseLU";
                return std::make_unique<EigenSparseLUSolver>();
        }
    }
    
    /**
     * @brief Create a solver by type string.
     * @param type Solver type string (e.g., "sparse_lu", "cg", "pardiso")
     */
    static std::unique_ptr<LinearSolver> create(const std::string& type) {
        return create(stringToSolverType(type));
    }
    
    /**
     * @brief Create a solver with configuration.
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
     */
    static std::unique_ptr<LinearSolver> create(
        const std::string& type,
        int maxIterations,
        Real tolerance,
        int printLevel = 0)
    {
        return create(stringToSolverType(type), maxIterations, tolerance, printLevel);
    }
    
    /**
     * @brief Get the best default solver for the system.
     * 
     * Priority:
     * 1. MKL PARDISO (if available)
     * 2. Eigen SparseLU
     */
    static std::unique_ptr<LinearSolver> createAuto() {
#ifdef MPFEM_USE_MKL
        LOG_DEBUG << "Auto-selecting PARDISO solver";
        return std::make_unique<PardisoSolver>();
#else
        LOG_DEBUG << "Auto-selecting Eigen SparseLU solver";
        return std::make_unique<EigenSparseLUSolver>();
#endif
    }
    
    /**
     * @brief Get a solver suitable for symmetric positive definite matrices.
     */
    static std::unique_ptr<LinearSolver> createForSPD(
        int maxIterations = 1000,
        Real tolerance = 1e-10,
        int printLevel = 0)
    {
        // For SPD matrices, CG with IC is usually best
        auto solver = std::make_unique<EigenCGICSolver>();
        solver->setMaxIterations(maxIterations);
        solver->setTolerance(tolerance);
        solver->setPrintLevel(printLevel);
        return solver;
    }
    
    /**
     * @brief Get a solver suitable for general non-symmetric matrices.
     */
    static std::unique_ptr<LinearSolver> createForGeneral(
        int maxIterations = 1000,
        Real tolerance = 1e-10,
        int printLevel = 0)
    {
        // For general matrices, use direct solver or BiCGSTAB
#ifdef MPFEM_USE_MKL
        auto solver = std::make_unique<PardisoSolver>();
#else
        auto solver = std::make_unique<EigenSparseLUSolver>();
#endif
        solver->setMaxIterations(maxIterations);
        solver->setTolerance(tolerance);
        solver->setPrintLevel(printLevel);
        return solver;
    }
    
    /**
     * @brief List available solvers.
     */
    static std::vector<std::string> availableSolvers() {
        std::vector<std::string> solvers = {
            "sparse_lu", "sparse_qr",
            "cg", "cg_ic",
            "bicgstab", "bicgstab_ilut",
            "auto"
        };
        
#ifdef MPFEM_USE_MKL
        solvers.push_back("pardiso");
#endif
        
        return solvers;
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_FACTORY_HPP

#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "model/field_kind.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "core/logger.hpp"
#include <memory>
#include <vector>
#include <map>
#include <deque>

namespace mpfem {

/**
 * @file coupling_manager.hpp
 * @brief Manager for multi-physics coupling.
 * 
 * Handles the iterative solution of coupled physics problems
 * using Picard iteration or Anderson acceleration.
 */

/**
 * @brief Iteration method for coupled problems.
 */
enum class IterationMethod {
    Picard,     ///< Fixed-point iteration (simple Picard)
    Anderson    ///< Anderson acceleration (improved convergence)
};

/**
 * @brief Result of a coupling iteration.
 */
struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
    Real electricPotentialError = 0.0;
    Real temperatureError = 0.0;
};

/**
 * @brief Manager for coupled multi-physics simulations.
 * 
 * Handles the iterative solution of coupled electro-thermal problems.
 * Supports two iteration methods:
 * - Picard: Simple fixed-point iteration
 * - Anderson: Anderson acceleration for faster convergence
 */
class CouplingManager {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    CouplingManager() = default;
    
    /**
     * @brief Construct with iteration method and parameters.
     * @param method Iteration method (Picard or Anderson)
     * @param maxIter Maximum number of iterations
     * @param tolerance Convergence tolerance (relative)
     * @param andersonDepth Number of previous iterations for Anderson (default: 5)
     */
    CouplingManager(IterationMethod method, int maxIter = 20, 
                    Real tolerance = 1e-8, int andersonDepth = 5);
    
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    
    /// Set iteration method
    void setMethod(IterationMethod method) { method_ = method; }
    
    /// Set maximum iterations
    void setMaxIterations(int maxIter) { maxIterations_ = maxIter; }
    
    /// Set convergence tolerance
    void setTolerance(Real tol) { tolerance_ = tol; }
    
    /// Set Anderson depth (number of previous iterations to use)
    void setAndersonDepth(int depth) { andersonDepth_ = depth; }
    
    /// Set relaxation factor for Picard iteration
    void setRelaxation(Real omega) { relaxation_ = omega; }
    
    // -------------------------------------------------------------------------
    // Solver Registration
    // -------------------------------------------------------------------------
    
    /**
     * @brief Register the electrostatics solver.
     */
    void setElectrostaticsSolver(ElectrostaticsSolver* solver) {
        electrostaticsSolver_ = solver;
    }
    
    /**
     * @brief Register the heat transfer solver.
     */
    void setHeatTransferSolver(HeatTransferSolver* solver) {
        heatTransferSolver_ = solver;
    }
    
    // -------------------------------------------------------------------------
    // Coupling Setup
    // -------------------------------------------------------------------------
    
    /**
     * @brief Enable Joule heating coupling.
     * @param enable Whether to enable Joule heating source
     */
    void enableJouleHeating(bool enable = true) {
        jouleHeatingEnabled_ = enable;
    }
    
    /**
     * @brief Enable temperature-dependent conductivity.
     * @param enable Whether to enable temperature-dependent conductivity
     */
    void enableTemperatureDependentConductivity(bool enable = true) {
        tempDependentConductivity_ = enable;
    }
    
    // -------------------------------------------------------------------------
    // Solve
    // -------------------------------------------------------------------------
    
    /**
     * @brief Solve the coupled problem.
     * @return CouplingResult with convergence information
     */
    CouplingResult solve();
    
    /**
     * @brief Perform a single coupling iteration.
     * @param iter Current iteration number
     * @return Relative error in this iteration
     */
    Real iterate(int iter);
    
    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------
    
    /// Get current iteration count
    int currentIteration() const { return currentIteration_; }
    
    /// Get current residual
    Real currentResidual() const { return currentResidual_; }
    
    /// Check if converged
    bool isConverged() const { return converged_; }
    
private:
    // -------------------------------------------------------------------------
    // Anderson Acceleration
    // -------------------------------------------------------------------------
    
    /**
     * @brief Apply Anderson acceleration to the temperature field.
     * @param current Current temperature solution
     * @param previous Previous iteration solutions
     * @return Accelerated solution
     */
    Vector applyAndersonAcceleration(const Vector& current,
                                     std::deque<Vector>& history);
    
    /**
     * @brief Solve the Anderson least-squares problem.
     * @param residuals Vector of residuals (G(x_k) - x_k)
     * @return Optimal combination weights
     */
    Vector solveAndersonLS(const std::deque<Vector>& residuals);
    
    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------
    
    /**
     * @brief Compute relative error between two vectors.
     */
    Real computeRelativeError(const Vector& current, const Vector& previous) const;
    
    /**
     * @brief Update Joule heating source in heat transfer solver.
     */
    void updateJouleHeating();
    
    // -------------------------------------------------------------------------
    // Member Variables
    // -------------------------------------------------------------------------
    
    IterationMethod method_ = IterationMethod::Picard;
    int maxIterations_ = 20;
    Real tolerance_ = 1e-8;
    int andersonDepth_ = 5;
    Real relaxation_ = 1.0;  // Under-relaxation factor (1.0 = no relaxation)
    
    ElectrostaticsSolver* electrostaticsSolver_ = nullptr;
    HeatTransferSolver* heatTransferSolver_ = nullptr;
    
    bool jouleHeatingEnabled_ = true;
    bool tempDependentConductivity_ = true;  // 默认启用温度相关电导率
    
    int currentIteration_ = 0;
    Real currentResidual_ = 0.0;
    bool converged_ = false;
    
    // Storage for Anderson acceleration
    std::deque<Vector> temperatureHistory_;
    std::deque<Vector> voltageHistory_;
    Vector previousTemperature_;
    Vector previousVoltage_;
};

// =============================================================================
// Inline Implementations
// =============================================================================

inline CouplingManager::CouplingManager(IterationMethod method, int maxIter,
                                         Real tolerance, int andersonDepth)
    : method_(method), maxIterations_(maxIter), tolerance_(tolerance),
      andersonDepth_(andersonDepth) {}

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP

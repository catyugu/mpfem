#ifndef MPFEM_TRANSIENT_PROBLEM_HPP
#define MPFEM_TRANSIENT_PROBLEM_HPP

#include "problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "time/time_scheme.hpp"
#include "core/logger.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Result of transient solve
 */
struct TransientResult {
    bool converged = false;
    int timeSteps = 0;
    Real finalTime = 0.0;
    
    std::vector<Real> times;                      ///< Time at each snapshot
    std::vector<FieldValues> snapshots;            ///< Field values at each time step
    
    void addSnapshot(Real time, const FieldValues& fields) {
        times.push_back(time);
        snapshots.emplace_back();
        snapshots.back() = fields;  // Copy
    }
    
    int numTimeSteps() const { return static_cast<int>(times.size()); }
};

class TransientProblem : public Problem {
public:
    bool isTransient() const override { return true; }
    
    Real startTime = 0.0;
    Real endTime = 1.0;
    Real timeStep = 0.01;
    Real currentTime = 0.0;
    int currentStep = 0;
    
    TimeScheme scheme = TimeScheme::BackwardEuler;
    
    void initializeTransient(int historyDepth = 2) {
        fieldValues.setMaxHistorySteps(historyDepth);
    }
    
    void advanceTime() {
        fieldValues.advanceTime();
        currentTime += timeStep;
        ++currentStep;
        LOG_INFO << "Time step " << currentStep << ", t = " << currentTime;
    }
    
    bool finished() const {
        return currentTime >= endTime - 1e-10;
    }
    
    const GridFunction& history(FieldId id, int stepsBack = 1) const {
        return fieldValues.history(id, stepsBack);
    }
    
    GridFunction& history(FieldId id, int stepsBack = 1) {
        return fieldValues.history(id, stepsBack);
    }
    
    /**
     * @brief Solve the transient problem with time stepping and Picard coupling
     * 
     * Outer loop: time stepping
     * Inner loop: Picard coupling iteration
     *   - Electrostatics: quasi-static, temperature-dependent conductivity
     *   - Heat: transient (handled by time integrator)
     *   - Structural: quasi-static, thermal stress
     */
    TransientResult solve();
    
private:
    Vector prevT_;  ///< Previous temperature for convergence check
    
    /// @brief Solve steady-state initialization at t=0 before time stepping
    void initializeSteadyState();
};

}  // namespace mpfem

#endif  // MPFEM_TRANSIENT_PROBLEM_HPP

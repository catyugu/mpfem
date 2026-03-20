#ifndef MPFEM_TRANSIENT_PROBLEM_HPP
#define MPFEM_TRANSIENT_PROBLEM_HPP

#include "problem.hpp"
#include "core/logger.hpp"

namespace mpfem {

/// Time integration scheme
enum class TimeScheme {
    BackwardEuler,   ///< First-order, unconditionally stable
    BDF2,            ///< Second-order backward differentiation
    CrankNicolson    ///< Second-order, unconditionally stable
};

/**
 * @brief Transient problem
 * 
 * Extends Problem with time dimension.
 * Uses FieldValues from base class for history field management.
 */
class TransientProblem : public Problem {
public:
    bool isTransient() const override { return true; }
    
    // Time parameters
    Real startTime = 0.0;
    Real endTime = 1.0;
    Real timeStep = 0.01;
    Real currentTime = 0.0;
    int currentStep = 0;
    
    /// Time integration scheme
    TimeScheme scheme = TimeScheme::BackwardEuler;
    
    /// Initialize transient problem (set history depth)
    void initializeTransient(int historyDepth = 2) {
        fieldValues.setMaxHistorySteps(historyDepth);
    }
    
    /// Advance one time step
    void advanceTime() {
        fieldValues.advanceTime();
        currentTime += timeStep;
        ++currentStep;
        LOG_INFO << "Time step " << currentStep << ", t = " << currentTime;
    }
    
    /// Check if simulation finished
    bool finished() const {
        return currentTime >= endTime - 1e-10;
    }
    
    /// Get history field (convenience wrapper)
    const GridFunction& history(FieldId id, int stepsBack = 1) const {
        return fieldValues.history(id, stepsBack);
    }
    
    GridFunction& history(FieldId id, int stepsBack = 1) {
        return fieldValues.history(id, stepsBack);
    }
};

}  // namespace mpfem

#endif  // MPFEM_TRANSIENT_PROBLEM_HPP

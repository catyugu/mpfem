#ifndef MPFEM_TRANSIENT_PROBLEM_HPP
#define MPFEM_TRANSIENT_PROBLEM_HPP

#include "problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

enum class TimeScheme {
    BackwardEuler,
    BDF2,
    CrankNicolson
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
};

}  // namespace mpfem

#endif  // MPFEM_TRANSIENT_PROBLEM_HPP

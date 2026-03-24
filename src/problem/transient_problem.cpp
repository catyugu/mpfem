#include "problem/transient_problem.hpp"
#include "time/time_integrator.hpp"
#include "core/logger.hpp"

namespace mpfem {

void TransientProblem::initializeSteadyState() {
    LOG_INFO << "Steady-state initialization at t=0";
    
    // 1. Electrostatics: solve with initial temperature
    if (hasElectrostatics()) {
        electrostatics->assemble();
        electrostatics->solve();
    }
    
    // 2. Structural: solve with initial temperature
    if (hasStructural()) {
        structural->assemble();
        structural->solve();
    }
    
    LOG_INFO << "Steady-state initialization complete";
}

TransientResult TransientProblem::solve() {
    ScopedTimer timer("Transient solve");
    TransientResult result;

    // Ensure enough history depth for the selected time scheme.
    const int requiredHistoryDepth = (scheme == TimeScheme::BDF2) ? 3 : 2;
    if (fieldValues.maxHistorySteps() < requiredHistoryDepth) {
        initializeTransient(requiredHistoryDepth);
    }

    currentTime = startTime;
    currentStep = 0;

    // Steady-state initialization at t=0
    initializeSteadyState();

    // Push initial conditions to history BEFORE first time step
    // This ensures BDF1 has proper T_prev available
    fieldValues.advanceTime();

    // Save initial snapshot (after steady-state initialization and history setup)
    result.addSnapshot(currentTime, fieldValues);

    // Create time integrator
    std::unique_ptr<TimeIntegrator> integrator(createTimeIntegrator(scheme));
    if (!integrator) {
        LOG_ERROR << "TransientProblem::solve: Failed to create time integrator for scheme";
        return result;
    }

    LOG_INFO << "Starting transient solve: t=[" << startTime << ", " << endTime
             << "], dt=" << timeStep << ", scheme=" << static_cast<int>(scheme);

    const Real eps = 1e-10;

    // Time stepping loop: one committed step per outer iteration.
    while (currentTime + timeStep <= endTime + eps) {
        const Real nextTime = currentTime + timeStep;
        LOG_INFO << "Time step " << (currentStep + 1) << ", t=" << nextTime;
        
        // Reset previous temperature for new time step
        prevT_.resize(0);

        // Picard coupling iteration within this time step (electrostatics + heat transfer only)
        // Structural (displacement) is uni-directionally coupled from temperature (thermal expansion)
        // Only compute after electro-thermal coupling converges
        bool couplingConverged = false;
        
        const bool hasElectrostatics = this->hasElectrostatics();
        const bool hasHeatTransfer = this->hasHeatTransfer();
        const bool hasStructural = this->hasStructural();
        
        for (int picardIter = 0; picardIter < couplingMaxIter; ++picardIter) {
            
            // 1. Electrostatics: quasi-static, depends on temperature via sigma(T)
            if (hasElectrostatics) {
                electrostatics->assemble();
                electrostatics->solve();
            }
            
            // 2. Heat transfer: TRANSIENT - use time integrator to advance
            if (hasHeatTransfer) {
                // Re-assemble heat solver to pick up any changed coefficients
                // (e.g., Joule heat from updated electrostatics solution)
                heatTransfer->assemble();
                
                // Use time integrator to do the transient heat solve
                bool stepOk = integrator->step(*this);
                if (!stepOk) {
                    LOG_ERROR << "TransientProblem::solve: Time step failed";
                    return result;
                }
            }
            
            // Check coupling convergence (based on temperature change between iterations)
            Real errT = 0.0;
            if (hasHeatTransfer) {
                const auto& T = heatTransfer->field().values();
                if (prevT_.size() == 0) {
                    prevT_ = T;
                    errT = 1.0;  // First iteration - don't converge yet
                } else {
                    errT = (T - prevT_).norm() / (T.norm() + 1e-15);
                    prevT_ = T;
                }
            }
            
            LOG_INFO << "  Picard iter " << (picardIter + 1) << ", T residual = " << errT;
            
            if (errT < couplingTol) {
                couplingConverged = true;
                break;
            }
        }
        
        // Structural: quasi-static (thermal stress), depends on temperature
        // Compute ONLY ONCE after electro-thermal coupling converges
        // (uni-directional coupling: temperature -> displacement, not vice versa)
        if (hasStructural) {
            structural->assemble();
            structural->solve();
        }
        
        if (!couplingConverged) {
            LOG_ERROR << "TransientProblem::solve: Coupling not converged at t=" << nextTime;
            return result;
        }
        
        // Commit this time step after coupling convergence.
        currentTime = nextTime;
        ++currentStep;

        // Save snapshot at this committed time point.
        result.addSnapshot(nextTime, fieldValues);
        
        // Push committed fields to history for the next time step.
        fieldValues.advanceTime();
    }

    result.timeSteps = currentStep;
    result.finalTime = currentTime;
    LOG_INFO << "Transient solve completed: " << result.timeSteps << " time steps";
    result.converged = true;
    
    return result;
}

}  // namespace mpfem

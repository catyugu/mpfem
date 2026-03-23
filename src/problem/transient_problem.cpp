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

    // Steady-state initialization at t=0
    initializeSteadyState();

    // Push initial conditions to history BEFORE first time step
    // This ensures BDF1 has proper T_prev available
    advanceTime();

    // Save t=0 snapshot (after steady-state initialization and history setup)
    result.addSnapshot(0.0, fieldValues);

    // Create time integrator
    auto* integrator = createTimeIntegrator(scheme);
    if (!integrator) {
        LOG_ERROR << "TransientProblem::solve: Failed to create time integrator for scheme";
        return result;
    }

    LOG_INFO << "Starting transient solve: t=[" << startTime << ", " << endTime 
             << "], dt=" << timeStep << ", scheme=" << static_cast<int>(scheme);

    // Time stepping loop
    while (!finished()) {
        LOG_INFO << "Time step " << (currentStep + 1) << ", t=" << (currentTime + timeStep);
        
        // Reset previous temperature for new time step
        prevT_.resize(0);

        // Picard coupling iteration within this time step
        bool couplingConverged = false;
        for (int picardIter = 0; picardIter < couplingMaxIter; ++picardIter) {
            
            // 1. Electrostatics: quasi-static, depends on temperature via sigma(T)
            if (hasElectrostatics()) {
                electrostatics->assemble();
                electrostatics->solve();
            }
            
            // 2. Heat transfer: TRANSIENT - use time integrator to advance
            if (hasHeatTransfer()) {
                // Re-assemble heat solver to pick up any changed coefficients
                // (e.g., Joule heat from updated electrostatics solution)
                heatTransfer->assemble();
                
                // Use time integrator to do the transient heat solve
                // Note: T_prev comes from history (properly set by advanceTime before loop)
                bool stepOk = integrator->step(*this);
                if (!stepOk) {
                    LOG_ERROR << "TransientProblem::solve: Time step failed";
                    delete integrator;
                    return result;
                }
            }
            
            // 3. Structural: quasi-static (thermal stress), depends on temperature
            if (hasStructural()) {
                structural->assemble();
                structural->solve();
            }
            
            // Check coupling convergence (based on temperature change between iterations)
            Real errT = 0.0;
            if (hasHeatTransfer()) {
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
        
        if (!couplingConverged) {
            LOG_ERROR << "TransientProblem::solve: Coupling not converged at t=" << (currentTime + timeStep);
            delete integrator;
            return result;
        }
        
        // Save snapshot after advancing time
        result.addSnapshot(currentTime + timeStep, fieldValues);
        
        // Advance time: push current fields to history for next time step
        advanceTime();
        result.timeSteps = currentStep;
        result.finalTime = currentTime;
    }
    
    LOG_INFO << "Transient solve completed: " << result.timeSteps << " time steps";
    result.converged = true;
    
    delete integrator;
    return result;
}

}  // namespace mpfem

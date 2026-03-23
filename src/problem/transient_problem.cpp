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

    // Initialize history storage for transient analysis
    initializeTransient(2);

    // Steady-state initialization at t=0
    initializeSteadyState();

    // Push initial conditions to history BEFORE first time step
    // This ensures BDF1 has proper T_prev available
    fieldValues.advanceTime();

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
    // We compute nextTime and check BEFORE solving, to ensure we save at all requested times
    int stepNum = 0;
    Real nextTime = timeStep;
    while (nextTime <= endTime + 1e-10) {
        ++stepNum;
        LOG_INFO << "Time step " << stepNum << ", t=" << nextTime;
        
        // Reset previous temperature for new time step
        prevT_.resize(0);

        // Picard coupling iteration within this time step (electrostatics + heat transfer only)
        // Structural (displacement) is uni-directionally coupled from temperature (thermal expansion)
        // Only compute after electro-thermal coupling converges
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
                bool stepOk = integrator->step(*this);
                if (!stepOk) {
                    LOG_ERROR << "TransientProblem::solve: Time step failed";
                    delete integrator;
                    return result;
                }
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
        
        // Structural: quasi-static (thermal stress), depends on temperature
        // Compute ONLY ONCE after electro-thermal coupling converges
        // (uni-directional coupling: temperature -> displacement, not vice versa)
        if (hasStructural()) {
            structural->assemble();
            structural->solve();
        }
        
        if (!couplingConverged) {
            LOG_ERROR << "TransientProblem::solve: Coupling not converged at t=" << nextTime;
            delete integrator;
            return result;
        }
        
        // Save snapshot at this time point
        result.addSnapshot(nextTime, fieldValues);
        
        // Push current fields to history for next time step
        fieldValues.advanceTime();
        
        // Advance to next time
        nextTime += timeStep;
    }
    
    result.timeSteps = stepNum;
    result.finalTime = endTime;
    LOG_INFO << "Transient solve completed: " << result.timeSteps << " time steps";
    result.converged = true;
    
    delete integrator;
    return result;
}

}  // namespace mpfem

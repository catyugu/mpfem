#include "time/bdf2_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF2Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        Vector historyCombo;

        if (currentStep > 0) {
            // Full BDF2 formula: historyCombo = 2*u_n - 0.5*u_{n-1}
            const GridFunction& prev1 = history.history(solver.fieldName(), 1);
            const GridFunction& prev2 = history.history(solver.fieldName(), 2);
            historyCombo = 2.0 * prev1.values() - 0.5 * prev2.values();
            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF2)";
        }
        else {
            // BDF1 starter for first step
            historyCombo = history.history(solver.fieldName(), 1).values();
            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF1 starter)";
        }

        // BDF2 formula: (1.5*M + dt*K) * u^{n+1} = M * (2*u^n - 0.5*u^{n-1}) + dt * F
        // => solveTransientStep(alpha=1.5, beta=dt, gamma=dt, historyRhs=2*u^n - 0.5*u^{n-1})
        if (!solver.solveTransientStep(1.5, dt, dt, historyCombo)) {
            LOG_ERROR << "BDF2Integrator: Transient solve failed for " << solver.fieldName();
            return false;
        }

        // Mark field updated - this is the BDF2 integrator's responsibility
        solver.field().markUpdated();

        LOG_INFO << "BDF2Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
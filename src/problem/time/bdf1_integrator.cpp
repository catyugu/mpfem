#include "problem/time/bdf1_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF1Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, Real prev_dt, int currentStep)
    {
        (void)currentStep;
        (void)prev_dt;

        // For BDF1: historyCombo is just the previous step value
        const GridFunction& prev = history.history(solver.fieldName(), 1);
        const Vector historyCombo = prev.values();

        // BDF1 formula: (1*M + dt*K) * u^{n+1} = M * u^n + dt * F
        // => solveTransientStep(alpha=1.0, beta=dt, gamma=dt, historyRhs=u^n)
        if (!solver.solveTransientStep(1.0, dt, dt, historyCombo)) {
            LOG_ERROR << "BDF1Integrator: Transient solve failed for " << solver.fieldName();
            return false;
        }

        // Mark field updated - this is applyEssentialBCsor's responsibility
        solver.field().markUpdated();

        LOG_INFO << "BDF1Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
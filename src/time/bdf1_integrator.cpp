#include "time/bdf1_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF1Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        (void)currentStep;

        // For BDF1: historyCombo is just the previous step value
        const GridFunction& prev = history.history(solver.fieldName(), 1);
        const Vector historyCombo = prev.values();

        // Delegate to solver's transient solve (handles M, K, A, RHS, BCs, caching)
        if (!solver.solveTransient(dt, historyCombo)) {
            LOG_ERROR << "BDF1Integrator: Transient solve failed for " << solver.fieldName();
            return false;
        }

        LOG_INFO << "BDF1Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
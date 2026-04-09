#include "time/bdf2_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF2Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, Real prev_dt, int currentStep)
    {
        Vector historyCombo;

        if (currentStep == 0 || prev_dt <= 0.0) {
            // BDF1 starter for first step or when prev_dt unavailable
            historyCombo = history.history(solver.fieldName(), 1).values();
            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF1 starter)";
            if (!solver.solveTransientStep(1.0, dt, dt, historyCombo)) {
                LOG_ERROR << "BDF2Integrator: Transient solve failed for " << solver.fieldName();
                return false;
            }
        }
        else {
            // Variable step size BDF2 coefficients
            // omega = dt / prev_dt
            Real omega = dt / prev_dt;
            Real alpha = (1.0 + 2.0 * omega) / (1.0 + omega);
            Real beta_n = (1.0 + omega);
            Real beta_nm1 = -(omega * omega) / (1.0 + omega);

            const GridFunction& prev1 = history.history(solver.fieldName(), 1);
            const GridFunction& prev2 = history.history(solver.fieldName(), 2);
            historyCombo = beta_n * prev1.values() + beta_nm1 * prev2.values();

            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1)
                     << " (omega=" << omega << ", alpha=" << alpha << ")";

            // BDF2 formula: (alpha*M + dt*K) * u^{n+1} = M * historyCombo + dt*F
            if (!solver.solveTransientStep(alpha, dt, dt, historyCombo)) {
                LOG_ERROR << "BDF2Integrator: Transient solve failed for " << solver.fieldName();
                return false;
            }
        }

        solver.field().markUpdated();
        LOG_INFO << "BDF2Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
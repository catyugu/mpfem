#include "time/bdf2_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF2Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        SparseMatrix M, K;
        Vector F;

        solver.buildMassMatrix(M);
        solver.buildStiffnessMatrix(K);
        solver.buildRHS(F);

        if (M.rows() == 0 || M.cols() == 0) {
            LOG_ERROR << "BDF2Integrator: Mass matrix not available for " << solver.fieldName();
            return false;
        }

        ensureSize(M.rows(), M.cols());

        GridFunction& curr = solver.field();

        if (currentStep > 0) {
            // Full BDF2 formula
            const GridFunction& Tprev1 = history.history(solver.fieldName(), 1);
            const GridFunction& Tprev2 = history.history(solver.fieldName(), 2);

            A_ = (1.5 * M) + (dt * K);
            A_.makeCompressed();

            const Vector historyCombo = 2.0 * Tprev1.values() - 0.5 * Tprev2.values();
            rhs_ = M * historyCombo + dt * F;

            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF2)";
        }
        else {
            // BDF1 starter for first step
            const GridFunction& Tprev = history.history(solver.fieldName(), 1);

            A_ = M + (dt * K);
            A_.makeCompressed();

            rhs_ = M * Tprev.values() + dt * F;

            LOG_INFO << "BDF2Integrator: Step " << (currentStep + 1) << " (using BDF1 starter)";
        }

        solver.applyBoundaryConditions(A_, rhs_, curr.values());

        if (!solver.solveLinearSystem(A_, curr.values(), rhs_)) {
            LOG_ERROR << "BDF2Integrator: Linear solve failed for " << solver.fieldName();
            return false;
        }

        LOG_INFO << "BDF2Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
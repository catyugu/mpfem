#include "time/bdf1_integrator.hpp"

#include "core/logger.hpp"

namespace mpfem {

    bool BDF1Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep)
    {
        (void)currentStep;

        SparseMatrix M, K;
        Vector F;

        solver.buildMassMatrix(M);
        solver.buildStiffnessMatrix(K);
        solver.buildRHS(F);

        if (M.rows() == 0 || M.cols() == 0) {
            LOG_ERROR << "BDF1Integrator: Mass matrix not available for " << solver.fieldName();
            return false;
        }

        ensureSize(M.rows(), M.cols());

        const GridFunction& prev = history.history(solver.fieldName(), 1);
        GridFunction& curr = solver.field();

        A_ = M + (dt * K);
        A_.makeCompressed();
        rhs_ = M * prev.values() + dt * F;

        solver.applyEssentialBCs(A_, rhs_, curr.values());

        if (!solver.solveLinearSystem(A_, curr.values(), rhs_)) {
            LOG_ERROR << "BDF1Integrator: Linear solve failed for " << solver.fieldName();
            return false;
        }

        LOG_INFO << "BDF1Integrator: Step completed for " << solver.fieldName()
                 << ", iterations: " << solver.iterations();
        return true;
    }

} // namespace mpfem
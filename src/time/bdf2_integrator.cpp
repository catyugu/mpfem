#include "time/bdf2_integrator.hpp"

#include "core/logger.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "problem/transient_problem.hpp"

namespace mpfem {

    bool BDF2Integrator::step(TransientProblem& problem)
    {
        if (!problem.heatTransfer) {
            LOG_ERROR << "BDF2Integrator: HeatTransfer solver not available";
            return false;
        }

        auto* heatSolver = problem.heatTransfer.get();
        if (!heatSolver->massMatrixAssembled()) {
            LOG_ERROR << "BDF2Integrator: Mass matrix not assembled";
            return false;
        }

        const SparseMatrix& M = heatSolver->massMatrix();
        const SparseMatrix& K = heatSolver->stiffnessMatrixBeforeBC();
        const Vector& Q = heatSolver->rhsBeforeBC();
        GridFunction& Tcurr = heatSolver->field();
        const Real dt = problem.timeStep;

        ensureSize(M.rows(), M.cols());

        if (problem.currentStep > 0) {
            const GridFunction& Tprev1 = problem.history("T", 1);
            const GridFunction& Tprev2 = problem.history("T", 2);

            A_ = (1.5 * M) + (dt * K);
            A_.makeCompressed();

            const Vector historyCombo = 2.0 * Tprev1.values() - 0.5 * Tprev2.values();
            rhs_ = M * historyCombo + dt * Q;

            LOG_INFO << "BDF2Integrator: Step " << (problem.currentStep + 1) << " (using BDF2)";
        }
        else {
            const GridFunction& Tprev = problem.history("T", 1);

            A_ = M + (dt * K);
            A_.makeCompressed();

            rhs_ = M * Tprev.values() + dt * Q;

            LOG_INFO << "BDF2Integrator: Step " << (problem.currentStep + 1)
                     << " (using BDF1 starter)";
        }

        if (!heatSolver->solveLinearSystem(A_, Tcurr.values(), rhs_)) {
            LOG_ERROR << "BDF2Integrator: Linear solve failed";
            return false;
        }

        LOG_INFO << "BDF2Integrator: Step completed, iterations: " << heatSolver->iterations();
        return true;
    }

} // namespace mpfem
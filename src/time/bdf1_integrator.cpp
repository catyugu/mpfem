#include "time/bdf1_integrator.hpp"

#include "core/logger.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "problem/transient_problem.hpp"

namespace mpfem {

    bool BDF1Integrator::step(TransientProblem& problem)
    {
        if (!problem.heatTransfer) {
            LOG_ERROR << "BDF1Integrator: HeatTransfer solver not available";
            return false;
        }

        auto* heatSolver = problem.heatTransfer.get();
        if (!heatSolver->massMatrixAssembled()) {
            LOG_ERROR << "BDF1Integrator: Mass matrix not assembled";
            return false;
        }

        const SparseMatrix& M = heatSolver->massMatrix();
        const SparseMatrix& K = heatSolver->stiffnessMatrixBeforeBC();
        const Vector& Q = heatSolver->rhsBeforeBC();
        GridFunction& Tcurr = heatSolver->field();
        const GridFunction& Tprev = problem.history("HeatTransfer", 1);
        const Real dt = problem.timeStep;

        ensureSize(M.rows(), M.cols());

        A_ = M + (dt * K);
        A_.makeCompressed();
        rhs_ = M * Tprev.values() + dt * Q;

        if (!heatSolver->solveLinearSystem(A_, Tcurr.values(), rhs_)) {
            LOG_ERROR << "BDF1Integrator: Linear solve failed";
            return false;
        }

        LOG_INFO << "BDF1Integrator: Step completed, iterations: " << heatSolver->iterations();
        return true;
    }

} // namespace mpfem
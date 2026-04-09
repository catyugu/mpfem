#include "problem/transient_problem.hpp"
#include "core/logger.hpp"
#include "time/time_integrator.hpp"

namespace mpfem {

    namespace {
        constexpr Real kResidualEps = 1e-15;
        constexpr Real kTimeEps = 1e-10;

        Real temperatureResidual(const HeatTransferSolver& solver, Vector& prevT, int picardIter)
        {
            const auto& T = solver.field().values();
            if (picardIter == 0) {
                prevT = T;
                return 1.0;
            }
            const Real residual = (T - prevT).norm() / (T.norm() + kResidualEps);
            prevT = T;
            return residual;
        }

        bool solveCouplingStep(TransientProblem& problem,
            TimeIntegrator& integrator,
            bool hasElectrostatics,
            bool hasHeatTransfer,
            Real& residual)
        {
            residual = 0.0;
            if (!hasHeatTransfer) {
                if (hasElectrostatics) {
                    SparseMatrix K;
                    Vector F;
                    problem.electrostatics->buildStiffnessMatrix(K);
                    problem.electrostatics->buildRHS(F);
                    problem.electrostatics->applyEssentialBCs(K, F, problem.electrostatics->field().values());
                    problem.electrostatics->solveLinearSystem(K, problem.electrostatics->field().values(), F);
                }
                return true;
            }

            Vector prevT;
            for (int picardIter = 0; picardIter < problem.couplingMaxIter; ++picardIter) {
                if (hasElectrostatics) {
                    SparseMatrix K;
                    Vector F;
                    problem.electrostatics->buildStiffnessMatrix(K);
                    problem.electrostatics->buildRHS(F);
                    problem.electrostatics->applyEssentialBCs(K, F, problem.electrostatics->field().values());
                    problem.electrostatics->solveLinearSystem(K, problem.electrostatics->field().values(), F);
                }

                if (!integrator.step(*problem.heatTransfer, problem.fieldValues, problem.timeStep, problem.currentStep)) {
                    LOG_ERROR << "TransientProblem::solve: Time step failed";
                    return false;
                }

                residual = temperatureResidual(*problem.heatTransfer, prevT, picardIter);
                LOG_INFO << "  Picard iter " << (picardIter + 1) << ", T residual = " << residual;

                if (residual < problem.couplingTol) {
                    return true;
                }
            }
            return false;
        }
    } // namespace

    void TransientProblem::initializeSteadyState()
    {
        LOG_INFO << "Steady-state initialization at t=0";

        if (hasElectrostatics()) {
            SparseMatrix K;
            Vector F;
            electrostatics->buildStiffnessMatrix(K);
            electrostatics->buildRHS(F);
            electrostatics->applyEssentialBCs(K, F, electrostatics->field().values());
            electrostatics->solveLinearSystem(K, electrostatics->field().values(), F);
        }

        if (hasStructural()) {
            SparseMatrix K;
            Vector F;
            structural->buildStiffnessMatrix(K);
            structural->buildRHS(F);
            structural->applyEssentialBCs(K, F, structural->field().values());
            structural->solveLinearSystem(K, structural->field().values(), F);
        }

        LOG_INFO << "Steady-state initialization complete";
    }

    TransientResult TransientProblem::solve()
    {
        ScopedTimer timer("Transient solve");
        TransientResult result;

        const bool hasElectrostatics = this->hasElectrostatics();
        const bool hasHeatTransfer = this->hasHeatTransfer();
        const bool hasStructural = this->hasStructural();

        const int requiredHistoryDepth = (scheme == TimeScheme::BDF2) ? 3 : 2;
        if (fieldValues.maxHistorySteps() < requiredHistoryDepth) {
            initializeTransient(requiredHistoryDepth);
        }

        currentTime = startTime;
        currentStep = 0;

        initializeSteadyState();
        fieldValues.advanceTime();
        result.addSnapshot(currentTime, fieldValues);

        std::unique_ptr<TimeIntegrator> integrator = createTimeIntegrator(scheme);
        if (!integrator) {
            LOG_ERROR << "TransientProblem::solve: Failed to create time integrator for scheme";
            return result;
        }

        LOG_INFO << "Starting transient solve: t=[" << startTime << ", " << endTime
                 << "], dt=" << timeStep << ", scheme=" << static_cast<int>(scheme);

        while (currentTime + timeStep <= endTime + kTimeEps) {
            const Real nextTime = currentTime + timeStep;
            LOG_INFO << "Time step " << (currentStep + 1) << ", t=" << nextTime;

            Real errT = 0.0;
            const bool couplingConverged = solveCouplingStep(*this, *integrator, hasElectrostatics, hasHeatTransfer, errT);
            if (!couplingConverged) {
                LOG_ERROR << "TransientProblem::solve: Coupling not converged at t=" << nextTime;
                return result;
            }

            if (hasStructural) {
                SparseMatrix K;
                Vector F;
                structural->buildStiffnessMatrix(K);
                structural->buildRHS(F);
                structural->applyEssentialBCs(K, F, structural->field().values());
                structural->solveLinearSystem(K, structural->field().values(), F);
            }

            currentTime = nextTime;
            ++currentStep;
            result.addSnapshot(nextTime, fieldValues);
            fieldValues.advanceTime();
        }

        result.timeSteps = currentStep;
        result.finalTime = currentTime;
        LOG_INFO << "Transient solve completed: " << result.timeSteps << " time steps";
        result.converged = true;

        return result;
    }

} // namespace mpfem

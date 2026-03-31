#include "problem/transient_problem.hpp"
#include "time/time_integrator.hpp"
#include "core/logger.hpp"

namespace mpfem
{

    void TransientProblem::initializeSteadyState()
    {
        LOG_INFO << "Steady-state initialization at t=0";

        if (hasElectrostatics())
        {
            electrostatics->assemble();
            electrostatics->solve();
        }

        if (hasStructural())
        {
            structural->assemble();
            structural->solve();
        }

        LOG_INFO << "Steady-state initialization complete";
    }

    TransientResult TransientProblem::solve()
    {
        ScopedTimer timer("Transient solve");
        TransientResult result;

        const int requiredHistoryDepth = (scheme == TimeScheme::BDF2) ? 3 : 2;
        if (fieldValues.maxHistorySteps() < requiredHistoryDepth)
        {
            initializeTransient(requiredHistoryDepth);
        }

        currentTime = startTime;
        currentStep = 0;

        initializeSteadyState();
        fieldValues.advanceTime();
        result.addSnapshot(currentTime, fieldValues);

        std::unique_ptr<TimeIntegrator> integrator = createTimeIntegrator(scheme);
        if (!integrator)
        {
            LOG_ERROR << "TransientProblem::solve: Failed to create time integrator for scheme";
            return result;
        }

        LOG_INFO << "Starting transient solve: t=[" << startTime << ", " << endTime
                 << "], dt=" << timeStep << ", scheme=" << static_cast<int>(scheme);

        const Real eps = 1e-10;

        while (currentTime + timeStep <= endTime + eps)
        {
            const Real nextTime = currentTime + timeStep;
            LOG_INFO << "Time step " << (currentStep + 1) << ", t=" << nextTime;

            bool couplingConverged = false;

            const bool hasElectrostatics = this->hasElectrostatics();
            const bool hasHeatTransfer = this->hasHeatTransfer();
            const bool hasStructural = this->hasStructural();

            for (int picardIter = 0; picardIter < couplingMaxIter; ++picardIter)
            {

                if (hasElectrostatics)
                {
                    electrostatics->assemble();
                    electrostatics->solve();
                }

                if (hasHeatTransfer)
                {
                    heatTransfer->assemble();
                    if (!integrator->step(*this))
                    {
                        LOG_ERROR << "TransientProblem::solve: Time step failed";
                        return result;
                    }
                }

                Real errT = 0.0;
                if (hasHeatTransfer)
                {
                    const auto &T = heatTransfer->field().values();
                    if (picardIter == 0)
                    {
                        prevT_ = T;
                        errT = 1.0; // First iteration - force continue
                    }
                    else
                    {
                        errT = (T - prevT_).norm() / (T.norm() + 1e-15);
                        prevT_ = T;
                    }
                }

                LOG_INFO << "  Picard iter " << (picardIter + 1) << ", T residual = " << errT;

                if (errT < couplingTol)
                {
                    couplingConverged = true;
                    break;
                }
            }

            if (hasStructural)
            {
                structural->assemble();
                structural->solve();
            }

            if (!couplingConverged)
            {
                LOG_ERROR << "TransientProblem::solve: Coupling not converged at t=" << nextTime;
                return result;
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

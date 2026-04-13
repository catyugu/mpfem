#include "problem/transient_problem.hpp"
#include "core/logger.hpp"
#include "problem/time/adaptive_time_stepper.hpp"
#include "problem/time/time_integrator.hpp"

namespace mpfem {

    void TransientProblem::initializeSteadyState()
    {
        LOG_INFO << "Steady-state initialization at t=0";

        if (hasElectrostatics())
            electrostatics->solveSteady();
        if (hasStructural())
            structural->solveSteady();

        LOG_INFO << "Steady-state initialization complete";
    }

    TransientResult TransientProblem::solve()
    {
        ScopedTimer timer("Transient solve");
        TransientResult result;

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
                 << "], sampleStep=" << timeStep << ", scheme=" << static_cast<int>(scheme);

        // Delegate to AdaptiveTimeStepper
        AdaptiveTimeStepper stepper(AdaptiveTimeStepperConfig {
            .sampleStep = timeStep,
            .maxDt = 10.0,
            .growFactor = 2.0});
        return stepper.solve(*this, *integrator);
    }

} // namespace mpfem

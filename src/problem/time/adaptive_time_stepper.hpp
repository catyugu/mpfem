#ifndef MPFEM_ADAPTIVE_TIME_STEPPER_HPP
#define MPFEM_ADAPTIVE_TIME_STEPPER_HPP

#include "core/logger.hpp"
#include "problem/time/time_stepper.hpp"
#include <algorithm>

namespace mpfem {

    struct AdaptiveTimeStepperConfig {
        Real sampleStep = 0.01; ///< Sampling interval
        Real maxDt = 10.0; ///< Maximum step size
        Real growFactor = 1.5; ///< Step growth factor on success
        Real shrinkFactor = 0.5; ///< Step shrink factor on failure
        Real minDt = 1e-6; ///< Minimum step size threshold
        Real residualEps = 1e-15; ///< Normalization for residual computation
        Real initialDt = 0.1; ///< Optional initial dt
    };

    class AdaptiveTimeStepper : public TimeStepper {
    public:
        explicit AdaptiveTimeStepper(const AdaptiveTimeStepperConfig& config)
            : cfg_(config) { }

        TransientResult solve(TransientProblem& problem, TimeIntegrator& integrator) override
        {
            TransientResult result;
            Real currentTime = problem.startTime;

            // Initial dt = sampleStep / 5 (as per specification)
            Real underlyingDt = cfg_.initialDt;
            Real prev_dt = underlyingDt;
            int stepCount = 0;

            problem.initializeSteadyState();
            problem.fieldValues.advanceTime();
            result.addSnapshot(currentTime, problem.fieldValues);

            Real nextSampleTime = currentTime + cfg_.sampleStep;
            constexpr Real kTimeEps = 1e-12;

            while (currentTime < problem.endTime - kTimeEps) {
                // Determine actual dt for this step
                Real actualDt = underlyingDt;
                bool isSamplingStep = false;

                // Cut only to hit exact sample point - does NOT affect underlyingDt
                if (currentTime + actualDt >= nextSampleTime - kTimeEps) {
                    actualDt = nextSampleTime - currentTime;
                    isSamplingStep = true;
                }

                LOG_INFO << "Step " << (stepCount + 1)
                         << ", t=" << (currentTime + actualDt)
                         << ", dt=" << actualDt
                         << " (underlying=" << underlyingDt << ")";

                Real errT = 0.0;
                bool converged = tryCouplingStep(problem, integrator, actualDt, prev_dt, errT);

                if (converged) {
                    if (problem.hasStructural()) {
                        problem.structural->solveSteady();
                    }

                    currentTime += actualDt;
                    prev_dt = actualDt;
                    stepCount++;
                    problem.fieldValues.advanceTime();

                    if (isSamplingStep) {
                        result.addSnapshot(currentTime, problem.fieldValues);
                        nextSampleTime += cfg_.sampleStep;
                    }

                    // Grow UNDERLYING dt on success - actualDt may have been cut for sampling
                    underlyingDt = std::min(underlyingDt * cfg_.growFactor, cfg_.maxDt);
                }
                else {
                    LOG_WARNING << "Convergence failed, shrinking time step.";
                    underlyingDt *= cfg_.shrinkFactor;
                    if (underlyingDt < cfg_.minDt) {
                        LOG_ERROR << "Time step too small. Aborting.";
                        break;
                    }
                }
            }

            result.timeSteps = stepCount;
            result.finalTime = currentTime;
            result.converged = (currentTime >= problem.endTime - kTimeEps);
            return result;
        }

    private:
        bool tryCouplingStep(TransientProblem& problem, TimeIntegrator& integrator,
            Real dt, Real prev_dt, Real& residual)
        {
            const bool hasE = problem.hasElectrostatics();
            const bool hasT = problem.hasHeatTransfer();

            if (!hasT) {
                if (hasE)
                    problem.electrostatics->solveSteady();
                return true;
            }

            Vector prevT;
            for (int iter = 0; iter < problem.couplingMaxIter; ++iter) {
                if (hasE)
                    problem.electrostatics->solveSteady();

                if (!integrator.step(*problem.heatTransfer, problem.fieldValues, dt, prev_dt, 1)) {
                    return false;
                }

                const auto& currentT = problem.heatTransfer->field().values();
                if (iter == 0) {
                    prevT = currentT;
                    continue;
                }

                residual = (currentT - prevT).norm() / (currentT.norm() + cfg_.residualEps);
                prevT = currentT;

                if (residual < problem.couplingTol)
                    return true;
            }
            return false;
        }

        AdaptiveTimeStepperConfig cfg_;
    };

} // namespace mpfem
#endif // MPFEM_ADAPTIVE_TIME_STEPPER_HPP

#ifndef MPFEM_ADAPTIVE_TIME_STEPPER_HPP
#define MPFEM_ADAPTIVE_TIME_STEPPER_HPP

#include "core/logger.hpp"
#include "time/time_stepper.hpp"
#include <algorithm>

namespace mpfem {

    class AdaptiveTimeStepper : public TimeStepper {
    public:
        AdaptiveTimeStepper(Real sampleStep, Real maxDt = 10.0, Real growFactor = 2.0, Real shrinkFactor = 0.5, Real minDt = 1e-6)
            : sampleStep_(sampleStep), maxDt_(maxDt), growFactor_(growFactor), shrinkFactor_(shrinkFactor), minDt_(minDt) { }

        TransientResult solve(TransientProblem& problem, TimeIntegrator& integrator) override
        {
            TransientResult result;
            Real currentTime = problem.startTime;

            // Initial dt = sampleStep / 5 (as per specification)
            Real dt = sampleStep_ / 5.0;
            Real prev_dt = dt;
            int stepCount = 0;

            problem.initializeSteadyState();
            problem.fieldValues.advanceTime();
            result.addSnapshot(currentTime, problem.fieldValues);

            Real nextSampleTime = currentTime + sampleStep_;
            constexpr Real kTimeEps = 1e-12;

            while (currentTime < problem.endTime - kTimeEps) {
                // Ensure exact sampling times - cut dt to hit exact sample point
                bool isSamplingStep = false;
                if (currentTime + dt >= nextSampleTime - kTimeEps) {
                    dt = nextSampleTime - currentTime;
                    isSamplingStep = true;
                }

                LOG_INFO << "Attempting step " << stepCount + 1
                         << ", t=" << currentTime + dt << ", dt=" << dt;

                Real errT = 0.0;
                bool converged = tryCouplingStep(problem, integrator, dt, prev_dt, errT);

                if (converged) {
                    if (problem.hasStructural()) {
                        problem.structural->solveSteady();
                    }

                    currentTime += dt;
                    prev_dt = dt;
                    stepCount++;
                    problem.fieldValues.advanceTime();

                    if (isSamplingStep) {
                        result.addSnapshot(currentTime, problem.fieldValues);
                        nextSampleTime += sampleStep_;
                    }

                    // Grow dt by factor on success, cap at maxDt_
                    dt = std::min(dt * growFactor_, maxDt_);
                }
                else {
                    LOG_WARN << "Convergence failed at dt=" << dt << ", shrinking time step.";
                    dt *= shrinkFactor_;
                    if (dt < minDt_) {
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
            constexpr Real kResidualEps = 1e-15;

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

                residual = (currentT - prevT).norm() / (currentT.norm() + kResidualEps);
                prevT = currentT;

                if (residual < problem.couplingTol)
                    return true;
            }
            return false;
        }

        Real sampleStep_;
        Real maxDt_;
        Real growFactor_;
        Real shrinkFactor_;
        Real minDt_;
    };

} // namespace mpfem
#endif // MPFEM_ADAPTIVE_TIME_STEPPER_HPP

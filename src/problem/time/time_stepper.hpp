#ifndef MPFEM_TIME_STEPPER_HPP
#define MPFEM_TIME_STEPPER_HPP

#include "problem/time/time_integrator.hpp"
#include "problem/transient_problem.hpp"


namespace mpfem {

    class TimeStepper {
    public:
        virtual ~TimeStepper() = default;
        virtual TransientResult solve(TransientProblem& problem, TimeIntegrator& integrator) = 0;
    };

} // namespace mpfem
#endif // MPFEM_TIME_STEPPER_HPP

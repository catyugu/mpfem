#ifndef MPFEM_TIME_INTEGRATOR_HPP
#define MPFEM_TIME_INTEGRATOR_HPP

#include "core/types.hpp"
#include "physics/field_values.hpp"
#include "physics/physics_field_solver.hpp"
#include "time/time_scheme.hpp"
#include <memory>

namespace mpfem {

    class TimeIntegrator {

    public:
        virtual ~TimeIntegrator() = default;

        virtual bool step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) = 0;
    };

    std::unique_ptr<TimeIntegrator> createTimeIntegrator(TimeScheme scheme);

} // namespace mpfem

#endif // MPFEM_TIME_INTEGRATOR_HPP
#ifndef MPFEM_BDF1_INTEGRATOR_HPP
#define MPFEM_BDF1_INTEGRATOR_HPP

#include "physics/field_values.hpp"
#include "physics/physics_field_solver.hpp"
#include "time_integrator.hpp"

namespace mpfem {

    class BDF1Integrator : public TimeIntegrator {
    public:
        BDF1Integrator() = default;

        bool step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) override;
    };

} // namespace mpfem

#endif // MPFEM_BDF1_INTEGRATOR_HPP
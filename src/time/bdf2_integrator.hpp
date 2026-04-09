#ifndef MPFEM_BDF2_INTEGRATOR_HPP
#define MPFEM_BDF2_INTEGRATOR_HPP

#include "physics/field_values.hpp"
#include "physics/physics_field_solver.hpp"
#include "time_integrator.hpp"

namespace mpfem {

    class BDF2Integrator : public TimeIntegrator {
    public:
        BDF2Integrator() = default;

        bool step(PhysicsFieldSolver& solver, FieldValues& history, Real dt, int currentStep) override;
    };

} // namespace mpfem

#endif // MPFEM_BDF2_INTEGRATOR_HPP
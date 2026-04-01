#ifndef MPFEM_BDF1_INTEGRATOR_HPP
#define MPFEM_BDF1_INTEGRATOR_HPP

#include "time_integrator.hpp"

namespace mpfem {

    class BDF1Integrator : public TimeIntegrator {
    public:
        BDF1Integrator() = default;

        bool step(TransientProblem& problem) override;
    };

} // namespace mpfem

#endif // MPFEM_BDF1_INTEGRATOR_HPP
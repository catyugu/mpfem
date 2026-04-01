#ifndef MPFEM_BDF2_INTEGRATOR_HPP
#define MPFEM_BDF2_INTEGRATOR_HPP

#include "time_integrator.hpp"

namespace mpfem {

    class BDF2Integrator : public TimeIntegrator {
    public:
        BDF2Integrator() = default;

        bool step(TransientProblem& problem) override;
    };

} // namespace mpfem

#endif // MPFEM_BDF2_INTEGRATOR_HPP
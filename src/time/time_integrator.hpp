#ifndef MPFEM_TIME_INTEGRATOR_HPP
#define MPFEM_TIME_INTEGRATOR_HPP

#include "core/types.hpp"
#include "time/time_scheme.hpp"

namespace mpfem {

class TransientProblem;

class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;
    
    virtual bool step(TransientProblem& problem) = 0;
};

TimeIntegrator* createTimeIntegrator(TimeScheme scheme);

}  // namespace mpfem

#endif  // MPFEM_TIME_INTEGRATOR_HPP

#ifndef MPFEM_BDF2_INTEGRATOR_HPP
#define MPFEM_BDF2_INTEGRATOR_HPP

#include "time_integrator.hpp"
#include "solver/sparse_matrix.hpp"

namespace mpfem {

/**
 * @brief BDF2 (Second-Order Backward Differentiation Formula) time integrator
 * 
 * Implements BDF2 with BDF1 as starter for the first step(s).
 * 
 * BDF2 scheme for heat equation:
 * ρCp * (3*T^{n+1} - 4*T^n + T^{n-1})/(2*dt) - ∇·(k∇T^{n+1}) = Q^{n+1}
 * 
 * Which discretizes to:
 * (3/2 * M + dt*K) * T^{n+1} = 2*M*T^n - 1/2*M*T^{n-1} + dt*Q
 * 
 * BDF1 (Backward Euler) starter for first step:
 * (M + dt*K) * T^{1} = M * T^0 + dt*Q
 */
class BDF2Integrator : public TimeIntegrator {
public:
    BDF2Integrator() = default;
    
    /**
     * @brief Perform one BDF2 time step
     * 
     * Uses BDF1 for the first step, then switches to BDF2.
     * 
     * @param problem TransientProblem containing heat transfer solver
     * @return true if step succeeded, false otherwise
     */
    bool step(TransientProblem& problem) override;
};

}  // namespace mpfem

#endif  // MPFEM_BDF2_INTEGRATOR_HPP
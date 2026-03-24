#ifndef MPFEM_BDF1_INTEGRATOR_HPP
#define MPFEM_BDF1_INTEGRATOR_HPP

#include "time_integrator.hpp"
#include "solver/sparse_matrix.hpp"

namespace mpfem {

/**
 * @brief BDF1 (Backward Euler) time integrator
 * 
 * Implements fully implicit first-order backward difference scheme for
 * transient heat transfer problems.
 * 
 * BDF1 scheme for heat equation:
 * ρCp * (T^{n+1} - T^n)/dt - ∇·(k∇T^{n+1}) = Q^{n+1}
 * 
 * Which discretizes to:
 * (M + dt*K) * T^{n+1} = M * T^n + dt * Q
 * 
 * where M is the mass matrix and K is the stiffness matrix.
 */
class BDF1Integrator : public TimeIntegrator {
public:
    BDF1Integrator() = default;
    
    /**
     * @brief Perform one BDF1 time step
     * 
     * @param problem TransientProblem containing heat transfer solver
     * @return true if step succeeded, false otherwise
     */
    bool step(TransientProblem& problem) override;

private:
    SparseMatrix A_;  ///< Pre-allocated system matrix (reused every step)
    Vector rhs_;      ///< Pre-allocated RHS vector (reused every step)
    bool initialized_ = false;  ///< Track if A_ and rhs_ are sized
};

}  // namespace mpfem

#endif  // MPFEM_BDF1_INTEGRATOR_HPP

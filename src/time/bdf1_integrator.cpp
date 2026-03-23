#include "time/bdf1_integrator.hpp"
#include "problem/transient_problem.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool BDF1Integrator::step(TransientProblem& problem) {
    // Get heat transfer solver
    if (!problem.heatTransfer) {
        LOG_ERROR << "BDF1Integrator: HeatTransfer solver not available";
        return false;
    }
    
    auto* heatSolver = problem.heatTransfer.get();
    
    // Check that mass matrix is assembled
    if (!heatSolver->massMatrixAssembled()) {
        LOG_ERROR << "BDF1Integrator: Mass matrix not assembled";
        return false;
    }
    
    const Real dt = problem.timeStep;
    
    // Get matrices and vectors BEFORE BCs are applied
    // (stiffnessMatrixBeforeBC and rhsBeforeBC are cached during assemble())
    const SparseMatrix& M = heatSolver->massMatrix();
    const SparseMatrix& K = heatSolver->stiffnessMatrixBeforeBC();
    const Vector& Q = heatSolver->rhsBeforeBC();
    
    // Get current field values T^n (solution goes here) and previous field values T^{n-1}
    // At first time step (currentStep == 0), history is not available yet,
    // so T^n is the current field (initial condition)
    GridFunction& T_curr = heatSolver->field();
    const GridFunction& T_prev = (problem.currentStep > 0) ? 
        problem.history(FieldId::Temperature, 1) : heatSolver->field();
    
    // Build effective stiffness matrix: A = M + dt * K
    // For BDF1 (Backward Euler), the implicit system is:
    // (M + dt*K) * T^{n+1} = M * T^n + dt * Q
    SparseMatrix A = M;
    A.eigen() = M.eigen() + dt * K.eigen();
    A.makeCompressed();
    
    // Compute RHS: rhs = M * T^n + dt * Q
    Vector rhs = M.eigen() * T_prev.values();
    rhs += dt * Q;
    
    // Solve A * T^{n+1} = rhs with boundary conditions applied to the combined system
    bool ok = heatSolver->solveLinearSystem(A, T_curr.values(), rhs);
    
    if (!ok) {
        LOG_ERROR << "BDF1Integrator: Linear solve failed";
        return false;
    }
    
    LOG_INFO << "BDF1Integrator: Step completed, iterations: " << heatSolver->iterations();
    return true;
}

}  // namespace mpfem

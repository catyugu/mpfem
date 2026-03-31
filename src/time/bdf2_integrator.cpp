#include "time/bdf2_integrator.hpp"
#include "problem/transient_problem.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool BDF2Integrator::step(TransientProblem& problem) {
    // Get heat transfer solver
    if (!problem.heatTransfer) {
        LOG_ERROR << "BDF2Integrator: HeatTransfer solver not available";
        return false;
    }
    
    auto* heatSolver = problem.heatTransfer.get();
    
    // Check that mass matrix is assembled
    if (!heatSolver->massMatrixAssembled()) {
        LOG_ERROR << "BDF2Integrator: Mass matrix not assembled";
        return false;
    }
    
    const Real dt = problem.timeStep;
    
    // Get matrices and vectors BEFORE BCs are applied
    const SparseMatrix& M = heatSolver->massMatrix();
    const SparseMatrix& K = heatSolver->stiffnessMatrixBeforeBC();
    const Vector& Q = heatSolver->rhsBeforeBC();
    
    // Get field values: T^{n+1} (current), T^n (history 1), T^{n-1} (history 2)
    GridFunction& T_curr = heatSolver->field();
    
    const Index nRows = M.rows();
    const Index nCols = M.cols();
    ensureSize(nRows, nCols);
    
    if (problem.currentStep == 0) {
        // First step: use BDF1 (Backward Euler)
        // (M + dt*K) * T^{n+1} = M * T^n + dt * Q
        const GridFunction& T_prev = problem.history(FieldId::Temperature, 1);  // T^n
        
        A_ = M + (dt * K);
        A_.makeCompressed();
        
        rhs_ = M * T_prev.values() + dt * Q;
        
        LOG_INFO << "BDF2Integrator: Step " << (problem.currentStep + 1)
                 << " (using BDF1 starter)";
    } else {
        // BDF2 formula:
        // (3/2 * M + dt*K) * T^{n+1} = 2*M*T^n - 1/2*M*T^{n-1} + dt*Q
        const GridFunction& T_prev1 = problem.history(FieldId::Temperature, 1);  // T^n
        const GridFunction& T_prev2 = problem.history(FieldId::Temperature, 2); // T^{n-1}
        
        A_ = (1.5 * M) + (dt * K);
        A_.makeCompressed();

        const Vector historyCombo = 2.0 * T_prev1.values() - 0.5 * T_prev2.values();
        rhs_ = M * historyCombo + dt * Q;
        
        LOG_INFO << "BDF2Integrator: Step " << (problem.currentStep + 1) << " (using BDF2)";
    }
    
    // Solve A * T^{n+1} = rhs with boundary conditions applied
    bool ok = heatSolver->solveLinearSystem(A_, T_curr.values(), rhs_);
    
    if (!ok) {
        LOG_ERROR << "BDF2Integrator: Linear solve failed";
        return false;
    }
    
    LOG_INFO << "BDF2Integrator: Step completed, iterations: " << heatSolver->iterations();
    return true;
}

}  // namespace mpfem
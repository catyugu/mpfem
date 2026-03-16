#ifndef MPFEM_SUPERLU_SOLVER_HPP
#define MPFEM_SUPERLU_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"

#ifdef MPFEM_USE_SUPERLU
#include <slu_Cnames.h>
#include <slu_ddefs.h>
#include <Eigen/SuperLUSupport>
#endif

namespace mpfem {

/**
 * @brief SuperLU direct solver using Eigen's native SuperLU support.
 * 
 * Zero-copy interface to SuperLU through Eigen's SuperLUSupport module.
 * Only available when MPFEM_USE_SUPERLU is defined.
 */
class SuperLUSolver : public LinearSolver {
public:
    SuperLUSolver() {
#ifndef MPFEM_USE_SUPERLU
        throw std::runtime_error("SuperLUSolver: SuperLU not available. "
                                 "Rebuild with -DMPFEM_USE_SUPERLU=ON");
#endif
    }
    
    ~SuperLUSolver() override = default;
    
    std::string name() const override { return "SuperLU"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
#ifdef MPFEM_USE_SUPERLU
        ScopedTimer timer("Linear solve (SuperLU)");
        
        // Get the Eigen sparse matrix (already compressed)
        const auto& mat = A.eigen();
        
        // Create SuperLU solver and factorize
        Eigen::SuperLU<Eigen::SparseMatrix<Real>> solver;
        solver.options().SymmetricMode = NO;
        solver.options().PrintStat = NO;
        solver.options().Equil = YES;
        
        solver.compute(mat);
        
        if (solver.info() != Eigen::Success) {
            LOG_ERROR << "SuperLU factorization failed";
            return false;
        }
        
        // Solve
        x = solver.solve(b);
        
        if (solver.info() != Eigen::Success) {
            LOG_ERROR << "SuperLU solve failed";
            return false;
        }
        
        iterations_ = 1;
        residual_ = 0.0;
        
        LOG_INFO << "[SuperLU] Solve successful, solution norm: " << x.norm();
        return true;
#else
        throw std::runtime_error("SuperLUSolver: SuperLU not available");
#endif
    }
};

}  // namespace mpfem

#endif  // MPFEM_SUPERLU_SOLVER_HPP
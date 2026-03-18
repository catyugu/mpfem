#ifndef MPFEM_UMFPACK_SOLVER_HPP
#define MPFEM_UMFPACK_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"
#include <vector>
#include <stdexcept>

#ifdef MPFEM_USE_SUITESPARSE
#include <Eigen/UmfPackSupport>
#endif

namespace mpfem {

#ifdef MPFEM_USE_SUITESPARSE

/**
 * @brief SuiteSparse UMFPACK direct solver.
 * 
 * High-performance direct LU solver from SuiteSparse.
 * Good alternative when MKL PARDISO is not available.
 */
class UmfpackSolver : public LinearSolver {
public:
    UmfpackSolver() = default;
    
    std::string name() const override { return "umfpack.lu"; }
    
    void setPrintLevel(int level) override {
        printLevel_ = level;
    }
    
    void analyzePattern(const SparseMatrix& A) override {
        ScopedTimer timer("UMFPACK symbolic analysis");
        
        solver_.analyzePattern(A.eigen());
        analyzed_ = (solver_.info() == Eigen::Success);
        
        if (!analyzed_) {
            LOG_ERROR << "UMFPACK symbolic analysis failed";
        } else if (printLevel_ >= 1) {
            LOG_INFO << "[UMFPACK] Symbolic analysis completed";
        }
    }
    
    void factorize(const SparseMatrix& A) override {
        ScopedTimer timer("UMFPACK factorization");
        
        if (!analyzed_) {
            analyzePattern(A);
        }
        
        solver_.factorize(A.eigen());
        factorized_ = (solver_.info() == Eigen::Success);
        
        if (!factorized_) {
            LOG_ERROR << "UMFPACK factorization failed";
        } else if (printLevel_ >= 1) {
            LOG_INFO << "[UMFPACK] Factorization completed";
        }
    }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (UMFPACK)");
        
        // UMFPACK requires re-factorization when matrix values change
        // (same pattern can be reused, but we need numerical factorization each time)
        solver_.analyzePattern(A.eigen());
        solver_.factorize(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "UMFPACK factorization failed";
            return false;
        }
        
        x = solver_.solve(b);
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "UMFPACK solve failed";
            return false;
        }
        
        analyzed_ = true;
        factorized_ = true;
        iterations_ = 1;
        residual_ = 0.0;
        
        LOG_INFO << "[UMFPACK] Solve successful, solution norm: " << x.norm();
        return true;
    }

private:
    Eigen::UmfPackLU<SparseMatrix::Storage> solver_;
    bool analyzed_ = false;
    bool factorized_ = false;
};

#else

// Stub for when SuiteSparse is not available
class UmfpackSolver : public LinearSolver {
public:
    UmfpackSolver() {
        throw std::runtime_error("UmfpackSolver: SuiteSparse not available. "
                                 "Rebuild with SuiteSparse support enabled.");
    }
    std::string name() const override { return "umfpack.lu"; }
    bool solve(const SparseMatrix&, Vector&, const Vector&) override { return false; }
};

#endif  // MPFEM_USE_SUITESPARSE

}  // namespace mpfem

#endif  // MPFEM_UMFPACK_SOLVER_HPP

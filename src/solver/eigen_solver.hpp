#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace mpfem {

/**
 * @brief Eigen SparseLU direct solver.
 * 
 * General-purpose sparse LU factorization for square matrices.
 * Suitable for both symmetric and non-symmetric systems.
 */
class EigenSparseLUSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::SparseLU"; }
    
    void analyzePattern(const SparseMatrix& A) override {
        solver_.analyzePattern(A.eigen());
    }
    
    void factorize(const SparseMatrix& A) override {
        solver_.factorize(A.eigen());
    }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (SparseLU)");
        
        // Analyze + factorize + solve in one call if not already done
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            std::cerr << "[EigenSparseLU] Factorization failed" << std::endl;
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solve(b);
        
        if (solver_.info() != Eigen::Success) {
            std::cerr << "[EigenSparseLU] Solve failed" << std::endl;
            x.setZero(b.size());
            return false;
        }
        
        LOG_INFO << "[EigenSparseLU] Solve successful, solution norm: " << x.norm();
        
        iterations_ = 1;
        residual_ = (A.eigen() * x - b).norm() / b.norm();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenSparseLU] Solved, residual = " << residual_;
        }
        
        return true;
    }
    
private:
    Eigen::SparseLU<Eigen::SparseMatrix<Real>> solver_;
};

/**
 * @brief Eigen CG with Incomplete Cholesky preconditioner.
 * 
 * For symmetric positive definite matrices.
 * Uses Incomplete Cholesky factorization as preconditioner for better convergence.
 */
class EigenCGICSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+IC"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (CG+IC)");
        
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.compute(A.eigen());
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenCG+IC] Iterations: " << iterations_ 
                     << ", Error: " << residual_;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>, Eigen::Lower, 
                             Eigen::IncompleteCholesky<Real>> solver_;
};

/**
 * @brief Eigen BiCGSTAB with ILUT preconditioner.
 * 
 * For non-symmetric matrices.
 * Uses Incomplete LU factorization with Threshold as preconditioner.
 */
class EigenBiCGSTABILUTSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::BiCGSTAB+ILUT"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (BiCGSTAB+ILUT)");
        
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.compute(A.eigen());
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenBiCGSTAB-ILUT] Iterations: " << iterations_ 
                     << ", Error: " << residual_;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real>, Eigen::IncompleteLUT<Real>> solver_;
};

// Note: GMRES is not available in standard Eigen, use BiCGSTAB or DGMRES from unsupported module
// If you need GMRES, consider using Eigen's unsupported GMRES or an external solver

}  // namespace mpfem

#endif  // MPFEM_EIGEN_SOLVER_HPP
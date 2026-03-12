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
 * @brief Eigen SparseQR direct solver.
 * 
 * Sparse QR factorization, works for rectangular matrices.
 * More numerically stable but slower than LU.
 */
class EigenSparseQRSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::SparseQR"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            if (printLevel_ > 0) {
                std::cerr << "[EigenSparseQR] Factorization failed" << std::endl;
            }
            return false;
        }
        
        x = solver_.solve(b);
        
        if (solver_.info() != Eigen::Success) {
            if (printLevel_ > 0) {
                std::cerr << "[EigenSparseQR] Solve failed" << std::endl;
            }
            return false;
        }
        
        iterations_ = 1;
        residual_ = (A.eigen() * x - b).norm() / b.norm();
        
        return true;
    }
    
private:
    Eigen::SparseQR<Eigen::SparseMatrix<Real>, Eigen::COLAMDOrdering<int>> solver_;
};

/**
 * @brief Eigen Conjugate Gradient iterative solver.
 * 
 * For symmetric positive definite matrices.
 */
class EigenCGSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenCG] Iterations: " << iterations_ 
                      << ", Error: " << residual_;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>> solver_;
};

/**
 * @brief Eigen CG with Incomplete Cholesky preconditioner.
 */
class EigenCGICSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+IC"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
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
 * @brief Eigen BiCGSTAB iterative solver.
 * 
 * For non-symmetric matrices.
 */
class EigenBiCGSTABSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::BiCGSTAB"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.compute(A.eigen());
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenBiCGSTAB] Iterations: " << iterations_ 
                     << ", Error: " << residual_;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real>> solver_;
};

/**
 * @brief Eigen BiCGSTAB with ILUT preconditioner.
 */
class EigenBiCGSTABILUTSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::BiCGSTAB+ILUT"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
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
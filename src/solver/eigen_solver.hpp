#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "linear_solver.hpp"
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
            if (printLevel_ > 0) {
                std::cerr << "[EigenSparseLU] Factorization failed" << std::endl;
            }
            return false;
        }
        
        x.eigen() = solver_.solve(b.eigen());
        
        if (solver_.info() != Eigen::Success) {
            if (printLevel_ > 0) {
                std::cerr << "[EigenSparseLU] Solve failed" << std::endl;
            }
            return false;
        }
        
        iterations_ = 1;
        residual_ = (A.eigen() * x.eigen() - b.eigen()).norm() / b.eigen().norm();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenSparseLU] Solved, residual = " << residual_ << std::endl;
        }
        
        return true;
    }
    
private:
    Eigen::SparseLU<SparseMatrix::Storage> solver_;
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
        
        x.eigen() = solver_.solve(b.eigen());
        
        if (solver_.info() != Eigen::Success) {
            if (printLevel_ > 0) {
                std::cerr << "[EigenSparseQR] Solve failed" << std::endl;
            }
            return false;
        }
        
        iterations_ = 1;
        residual_ = (A.eigen() * x.eigen() - b.eigen()).norm() / b.eigen().norm();
        
        return true;
    }
    
private:
    Eigen::SparseQR<SparseMatrix::Storage, Eigen::COLAMDOrdering<int>> solver_;
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
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenCG] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::ConjugateGradient<SparseMatrix::Storage> solver_;
};

/**
 * @brief Eigen CG with Incomplete Cholesky preconditioner.
 */
class EigenCGICSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+IC"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        // Create preconditioner
        precond_.compute(A.eigen());
        
        solver_.setPreconditioner(precond_);
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.compute(A.eigen());
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenCG-IC] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::ConjugateGradient<SparseMatrix::Storage, Eigen::Lower, 
                             Eigen::IncompleteCholesky<Real>> solver_;
    Eigen::IncompleteCholesky<Real> precond_;
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
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenBiCGSTAB] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::BiCGSTAB<SparseMatrix::Storage> solver_;
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
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenBiCGSTAB-ILUT] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::BiCGSTAB<SparseMatrix::Storage, Eigen::IncompleteLUT<Real>> solver_;
};

/**
 * @brief Eigen GMRES iterative solver.
 */
class EigenGMRESSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::GMRES"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.set_restart(30);  // Restart parameter
        solver_.compute(A.eigen());
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenGMRES] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::GMRES<SparseMatrix::Storage> solver_;
};

/**
 * @brief Eigen GMRES with ILUT preconditioner.
 */
class EigenGMRESILUTSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::GMRES+ILUT"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        solver_.set_restart(30);
        solver_.compute(A.eigen());
        
        x.eigen() = solver_.solveWithGuess(b.eigen(), x.eigen());
        
        iterations_ = solver_.iterations();
        residual_ = solver_.error();
        
        if (printLevel_ > 0) {
            std::cout << "[EigenGMRES-ILUT] Iterations: " << iterations_ 
                      << ", Error: " << residual_ << std::endl;
        }
        
        return solver_.info() == Eigen::Success;
    }
    
private:
    Eigen::GMRES<SparseMatrix::Storage, Eigen::IncompleteLUT<Real>> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_EIGEN_SOLVER_HPP

#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

/**
 * @brief Eigen CG with diagonal (Jacobi) preconditioner.
 * 
 * Recommended for well-conditioned symmetric positive-definite systems.
 * Cheap and effective for scalar Poisson/Laplace problems.
 */
class EigenCGJacobiSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+Jacobi"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (CG+Jacobi)");
        
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenCGJacobi] Setup failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        
        const bool success = solver_.info() == Eigen::Success;
        
        if (printLevel_ > 0 || !success) {
            LOG_INFO << "[EigenCGJacobi] Iterations: " << iterations_ 
                     << ", Error: " << residual_
                     << ", Success: " << (success ? "yes" : "no");
        }
        
        if (success) {
            LOG_INFO << "[EigenCGJacobi] Solve successful, solution norm: " << x.norm();
        }
        
        return success;
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Real>> solver_;
};

/**
 * @brief Eigen CG with Incomplete Cholesky (IC0) preconditioner.
 * 
 * Strong preconditioner for SPD systems, especially effective for
 * elasticity and ill-conditioned problems. More robust than Jacobi
 * for block-coupled systems.
 * 
 * Note: IncompleteCholesky uses shift parameter for regularization
 * on near-singular systems (default 1e-14).
 */
class EigenCGICCSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+ICC"; }
    
    /// Set shift parameter for regularization (default 1e-14)
    void setShift(Real shift) { shift_ = shift; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (CG+ICC)");
        
        // Configure solver
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenCGICC] Preconditioner setup failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        
        const bool success = solver_.info() == Eigen::Success;
        
        if (printLevel_ > 0 || !success) {
            LOG_INFO << "[EigenCGICC] Iterations: " << iterations_ 
                     << ", Error: " << residual_
                     << ", Success: " << (success ? "yes" : "no");
        }
        
        if (success) {
            LOG_INFO << "[EigenCGICC] Solve successful, solution norm: " << x.norm();
        }
        
        return success;
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteCholesky<Real>> solver_;
    Real shift_ = 1e-14;  // Regularization shift
};

/**
 * @brief Eigen CG with ILU preconditioner.
 * 
 * General-purpose preconditioner for moderately ill-conditioned SPD systems.
 * ILU is more robust than ICC for non-SPD matrices but less efficient for SPD.
 */
class EigenCGILUSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::CG+ILU"; }
    
    void setDropTolerance(Real tol) { dropTol_ = tol; }
    void setFillFactor(int fill) { fillFactor_ = fill; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (CG+ILU)");
        
        // Configure preconditioner
        solver_.preconditioner().setDroptol(dropTol_);
        solver_.preconditioner().setFillfactor(fillFactor_);
        
        // Configure solver
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenCGILU] Preconditioner setup failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        
        const bool success = solver_.info() == Eigen::Success;
        
        if (printLevel_ > 0 || !success) {
            LOG_INFO << "[EigenCGILU] Iterations: " << iterations_ 
                     << ", Error: " << residual_
                     << ", Success: " << (success ? "yes" : "no");
        }
        
        if (success) {
            LOG_INFO << "[EigenCGILU] Solve successful, solution norm: " << x.norm();
        }
        
        return success;
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteLUT<Real>> solver_;
    Real dropTol_ = 1e-4;
    int fillFactor_ = 10;
};

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
        
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenSparseLU] Factorization failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solve(b);
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenSparseLU] Solve failed";
            x.setZero(b.size());
            return false;
        }
        
        LOG_INFO << "[EigenSparseLU] Solve successful, solution norm: " << x.norm();
        
        iterations_ = 1;
        residual_ = (A.eigen() * x - b).norm() / b.norm();
        
        if (printLevel_ > 0) {
            LOG_INFO << "[EigenSparseLU] Residual = " << residual_;
        }
        
        return true;
    }
    
private:
    Eigen::SparseLU<Eigen::SparseMatrix<Real>> solver_;
};

/**
 * @brief Eigen DGMRES with ILU preconditioner.
 * 
 * Deflated GMRES with Incomplete LU factorization preconditioner.
 * Suitable for non-symmetric and indefinite systems (e.g., convection-diffusion).
 */
class EigenDGMRESILUSolver : public LinearSolver {
public:
    std::string name() const override { return "Eigen::DGMRES+ILU"; }
    
    void setDropTolerance(Real tol) { dropTol_ = tol; }
    void setFillFactor(int fill) { fillFactor_ = fill; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (DGMRES+ILU)");
        
        // Configure preconditioner
        solver_.preconditioner().setDroptol(dropTol_);
        solver_.preconditioner().setFillfactor(fillFactor_);
        
        // Configure solver
        solver_.setMaxIterations(maxIterations_);
        solver_.setTolerance(tolerance_);
        
        solver_.compute(A.eigen());
        
        if (solver_.info() != Eigen::Success) {
            LOG_ERROR << "[EigenDGMRES] Preconditioner setup failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver_.solveWithGuess(b, x);
        
        iterations_ = static_cast<int>(solver_.iterations());
        residual_ = solver_.error();
        
        const bool success = solver_.info() == Eigen::Success;
        
        if (printLevel_ > 0 || !success) {
            LOG_INFO << "[EigenDGMRES] Iterations: " << iterations_ 
                     << ", Error: " << residual_
                     << ", Success: " << (success ? "yes" : "no");
        }
        
        if (success) {
            LOG_INFO << "[EigenDGMRES] Solve successful, solution norm: " << x.norm();
        }
        
        return success;
    }
    
private:
    Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::IncompleteLUT<Real>> solver_;
    Real dropTol_ = 1e-4;
    int fillFactor_ = 10;
};

}  // namespace mpfem

#endif  // MPFEM_EIGEN_SOLVER_HPP
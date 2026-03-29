#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "linear_solver.hpp"
#include "core/logger.hpp"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

/**
 * @brief Base class for Eigen iterative solvers with common solve pattern.
 * 
 * Provides protected template method solveIterative() that handles the
 * boilerplate of iterative solver invocation, logging, and error handling.
 */
class EigenIterativeSolverBase : public LinearSolver {
protected:
    /**
     * @brief Template method for iterative solvers.
     * 
     * @tparam SolverType The Eigen solver type (e.g., ConjugateGradient, DGMRES)
     * @tparam HasILUPreconditioner Compile-time flag for ILU preconditioner config
     * @param solver The solver instance
     * @param A The sparse matrix
     * @param x Solution vector (modified in place)
     * @param b Right-hand side vector
     * @param solverName Name for logging
     * @return true if solve converged
     */
    template<typename SolverType, bool HasILUPreconditioner>
    bool solveIterative(SolverType& solver, const SparseMatrix& A, Vector& x, const Vector& b,
                        const std::string& solverName) {
        ScopedTimer timer("Linear solve (" + solverName + ")");
        
        // Preconditioner config (only ILU solvers) - compile-time dispatch via if constexpr
        if constexpr (HasILUPreconditioner) {
            solver.preconditioner().setDroptol(dropTol_);
            solver.preconditioner().setFillfactor(fillFactor_);
        }
        
        // Configure solver
        solver.setMaxIterations(maxIterations_);
        solver.setTolerance(tolerance_);
        
        solver.compute(A.eigen());
        
        if (solver.info() != Eigen::Success) {
            LOG_ERROR << "[" << solverName << "] Preconditioner setup failed";
            x.setZero(b.size());
            return false;
        }
        
        x = solver.solveWithGuess(b, x);
        
        iterations_ = static_cast<int>(solver.iterations());
        residual_ = solver.error();
        
        const bool success = solver.info() == Eigen::Success;
        
        if (printLevel_ > 0 || !success) {
            LOG_INFO << "[" << solverName << "] Iterations: " << iterations_
                     << ", Error: " << residual_
                     << ", Success: " << (success ? "yes" : "no");
        }
        
        if (success) {
            LOG_INFO << "[" << solverName << "] Solve successful, solution norm: " << x.norm();
        }
        
        return success;
    }
    
    Real dropTol_ = 1e-4;
    int fillFactor_ = 10;
};

/**
 * @brief Eigen CG with diagonal (Jacobi) preconditioner.
 * 
 * Recommended for well-conditioned symmetric positive-definite systems.
 * Cheap and effective for scalar Poisson/Laplace problems.
 */
class EigenCGJacobiSolver : public EigenIterativeSolverBase {
public:
    std::string name() const override { return "Eigen::CG+Jacobi"; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        return solveIterative<Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                                                         Eigen::Lower|Eigen::Upper,
                                                         Eigen::DiagonalPreconditioner<Real>>, false>(
            solver_, A, x, b, "CG+Jacobi");
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
class EigenCGICCSolver : public EigenIterativeSolverBase {
public:
    std::string name() const override { return "Eigen::CG+ICC"; }
    
    void applyConfig(const SolverConfig& config) override {
        if (config.dropTolerance > 0) setShift(config.dropTolerance);
        else setShift(1e-14);
    }
    
    /// Set shift parameter for regularization (default 1e-14)
    void setShift(Real shift) { shift_ = shift; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        ScopedTimer timer("Linear solve (CG+ICC)");
        
        return solveIterative<Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                                                         Eigen::Lower|Eigen::Upper,
                                                         Eigen::IncompleteCholesky<Real>>, false>(
            solver_, A, x, b, "CG+ICC");
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
class EigenCGILUSolver : public EigenIterativeSolverBase {
public:
    std::string name() const override { return "Eigen::CG+ILU"; }
    
    void applyConfig(const SolverConfig& config) override {
        setDropTolerance(config.dropTolerance);
        setFillFactor(config.fillFactor);
    }
    
    void setDropTolerance(Real tol) { dropTol_ = tol; }
    void setFillFactor(int fill) { fillFactor_ = fill; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        return solveIterative<Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                                                         Eigen::Lower|Eigen::Upper,
                                                         Eigen::IncompleteLUT<Real>>, true>(
            solver_, A, x, b, "CG+ILU");
    }
    
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteLUT<Real>> solver_;
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
class EigenDGMRESILUSolver : public EigenIterativeSolverBase {
public:
    std::string name() const override { return "Eigen::DGMRES+ILU"; }
    
    void applyConfig(const SolverConfig& config) override {
        setDropTolerance(config.dropTolerance);
        setFillFactor(config.fillFactor);
    }
    
    void setDropTolerance(Real tol) { dropTol_ = tol; }
    void setFillFactor(int fill) { fillFactor_ = fill; }
    
    bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override {
        return solveIterative<Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::IncompleteLUT<Real>>, true>(
            solver_, A, x, b, "DGMRES+ILU");
    }
    
private:
    Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::IncompleteLUT<Real>> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_EIGEN_SOLVER_HPP
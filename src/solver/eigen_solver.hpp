#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "core/logger.hpp"
#include "linear_solver.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <cstdint>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

    /**
     * @brief Base class for Eigen iterative solvers.
     *
     * Provides common infrastructure for CG and DGMRES solvers.
     */
    class EigenIterativeSolverBase : public LinearSolver {
    protected:
        Real dropTol_ = 1e-4;
        int fillFactor_ = 10;
    };

    /**
     * @brief Eigen CG solver.
     *
     * Uses Eigen's ConjugateGradient with diagonal preconditioning.
     */
    class EigenCGSolver : public EigenIterativeSolverBase {
    public:
        ~EigenCGSolver() override = default;

        std::string name() const override { return "Eigen::CG"; }

        bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override
        {
            ScopedTimer timer("Linear solve (CG)");

            solver_.setMaxIterations(maxIterations_);
            solver_.setTolerance(tolerance_);

            solver_.compute(A.eigen());

            if (solver_.info() != Eigen::Success) {
                LOG_ERROR << "[CG] Preconditioner setup failed";
                x.setZero(b.size());
                return false;
            }

            x = solver_.solveWithGuess(b, x);

            iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();

            const bool success = solver_.info() == Eigen::Success;

            if (printLevel_ > 0 || !success) {
                LOG_INFO << "[CG] Iterations: " << iterations_
                         << ", Error: " << residual_
                         << ", Success: " << (success ? "yes" : "no");
            }

            if (success) {
                LOG_INFO << "[CG] Solve successful, solution norm: " << x.norm();
            }

            return success;
        }

    private:
        Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Real>>
            solver_;
    };

    /**
     * @brief Eigen DGMRES solver.
     *
     * Uses Eigen's DGMRES with diagonal preconditioning.
     */
    class EigenDGMRESSolver : public EigenIterativeSolverBase {
    public:
        ~EigenDGMRESSolver() override = default;

        std::string name() const override { return "Eigen::DGMRES"; }

        bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override
        {
            ScopedTimer timer("Linear solve (DGMRES)");

            solver_.setMaxIterations(maxIterations_);
            solver_.setTolerance(tolerance_);

            solver_.compute(A.eigen());

            if (solver_.info() != Eigen::Success) {
                LOG_ERROR << "[DGMRES] Preconditioner setup failed";
                x.setZero(b.size());
                return false;
            }

            x = solver_.solveWithGuess(b, x);

            iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();

            const bool success = solver_.info() == Eigen::Success;

            if (printLevel_ > 0 || !success) {
                LOG_INFO << "[DGMRES] Iterations: " << iterations_
                         << ", Error: " << residual_
                         << ", Success: " << (success ? "yes" : "no");
            }

            if (success) {
                LOG_INFO << "[DGMRES] Solve successful, solution norm: " << x.norm();
            }

            return success;
        }

    private:
        Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::DiagonalPreconditioner<Real>> solver_;
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

        bool solve(const SparseMatrix& A, Vector& x, const Vector& b) override
        {
            ScopedTimer timer("Linear solve (SparseLU)");

            const std::uint64_t currentFingerprint = A.fingerprint();
            const bool needRefactor = !hasFactorCache_ || (currentFingerprint != lastMatrixFingerprint_);

            if (needRefactor) {
                solver_.compute(A.eigen());

                if (solver_.info() != Eigen::Success) {
                    LOG_ERROR << "[EigenSparseLU] Factorization failed";
                    x.setZero(b.size());
                    hasFactorCache_ = false;
                    return false;
                }

                hasFactorCache_ = true;
                lastMatrixFingerprint_ = currentFingerprint;
            }
            else if (printLevel_ > 0) {
                LOG_INFO << "[EigenSparseLU] Reusing cached factorization";
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
        bool hasFactorCache_ = false;
        std::uint64_t lastMatrixFingerprint_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_EIGEN_SOLVER_HPP
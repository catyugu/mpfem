#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "core/logger.hpp"
#include "linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

    // =============================================================================
    // Eigen CG Operator
    // =============================================================================

    class CgOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "CG"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("CgOperator: null matrix in setup");
            }
            solver_.setMaxIterations(maxIterations_);
            solver_.setTolerance(tolerance_);
            solver_.compute(A->eigen());
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solveWithGuess(b, x);
            iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        void set_max_iterations(int iter) { maxIterations_ = iter; }
        void set_tolerance(Real tol) { tolerance_ = tol; }
        int max_iterations() const { return maxIterations_; }
        Real tolerance() const { return tolerance_; }

        void configure(const LinearOperatorConfig& config) override
        {
            if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
                set_max_iterations(static_cast<int>(it->second));
            }
            if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
                set_tolerance(it->second);
            }
        }

        int iterations() const override { return iterations_; }
        Real residual() const override { return residual_; }

    private:
        int maxIterations_ = 1000;
        Real tolerance_ = 1e-10;
        int iterations_ = 0;
        Real residual_ = 0.0;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Real>>
            solver_;
    };

    // =============================================================================
    // Eigen DGMRES Operator
    // =============================================================================

    class GmresOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "DGMRES"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("GmresOperator: null matrix in setup");
            }
            solver_.setMaxIterations(maxIterations_);
            solver_.setTolerance(tolerance_);
            solver_.compute(A->eigen());
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solveWithGuess(b, x);
            iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        void set_max_iterations(int iter) { maxIterations_ = iter; }
        void set_tolerance(Real tol) { tolerance_ = tol; }
        int max_iterations() const { return maxIterations_; }
        Real tolerance() const { return tolerance_; }

        void configure(const LinearOperatorConfig& config) override
        {
            if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
                set_max_iterations(static_cast<int>(it->second));
            }
            if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
                set_tolerance(it->second);
            }
        }

        int iterations() const override { return iterations_; }
        Real residual() const override { return residual_; }

    private:
        int maxIterations_ = 1000;
        Real tolerance_ = 1e-10;
        int iterations_ = 0;
        Real residual_ = 0.0;
        Eigen::DGMRES<Eigen::SparseMatrix<Real>, Eigen::DiagonalPreconditioner<Real>> solver_;
    };

    // =============================================================================
    // Eigen SparseLU Operator
    // =============================================================================

    class EigenSparseLUOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "SparseLU"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("EigenSparseLUOperator: null matrix in setup");
            }
            const std::uint64_t currentFingerprint = A->fingerprint();
            if (!hasFactorCache_ || currentFingerprint != lastMatrixFingerprint_) {
                solver_.compute(A->eigen());
                if (solver_.info() != Eigen::Success) {
                    throw std::runtime_error("EigenSparseLUOperator: factorization failed");
                }
                hasFactorCache_ = true;
                lastMatrixFingerprint_ = currentFingerprint;
            }
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solve(b);
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("EigenSparseLUOperator: solve failed");
            }
        }

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }

    private:
        Eigen::SparseLU<Eigen::SparseMatrix<Real>> solver_;
        bool hasFactorCache_ = false;
        std::uint64_t lastMatrixFingerprint_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_EIGEN_SOLVER_HPP
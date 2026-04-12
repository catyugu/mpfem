#ifndef MPFEM_EIGEN_SOLVER_HPP
#define MPFEM_EIGEN_SOLVER_HPP

#include "core/logger.hpp"
#include "eigen_preconditioner_adapter.hpp"
#include "linear_operator.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

namespace mpfem {

    /**
     * @brief Unified template for Eigen iterative solvers.
     *
     * This class implements the DRY principle by providing a common base for
     * CG, DGMRES, BiCGSTAB, etc. It dynamically bridges our nested LinearOperator
     * preconditioners into Eigen's iterative solver loop.
     */
    template <typename EigenSolverType, const char* SolverName>
    class EigenIterativeOperator : public LinearOperator {
    public:
        std::string_view name() const override { return SolverName; }

        void setup(const SparseMatrix* A) override
        {
            if (!A)
                throw std::runtime_error(std::string(SolverName) + ": null matrix in setup");

            solver_.setMaxIterations(maxIterations_);
            solver_.setTolerance(tolerance_);
            solver_.compute(A->eigen());

            // Dynamically mount and setup the configured nested preconditioner
            if (preconditioner()) {
                preconditioner()->setup(A);
                solver_.preconditioner().set_operator(preconditioner());
            }
            else {
                solver_.preconditioner().set_operator(nullptr);
            }

            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solveWithGuess(b, x);
            iterations_ = static_cast<int>(solver_.iterations());
            residual_ = solver_.error();
        }

        void configure(const LinearOperatorConfig& config) override
        {
            if (auto it = config.parameters.find("MaxIterations"); it != config.parameters.end()) {
                maxIterations_ = static_cast<int>(it->second);
            }
            if (auto it = config.parameters.find("Tolerance"); it != config.parameters.end()) {
                tolerance_ = it->second;
            }
        }

        int iterations() const override { return iterations_; }
        Real residual() const override { return residual_; }

    private:
        int maxIterations_ = 1000;
        Real tolerance_ = 1e-10;
        int iterations_ = 0;
        Real residual_ = 0.0;
        EigenSolverType solver_;
    };

    // =============================================================================
    // Iterative Operator Definitions
    // =============================================================================

    inline constexpr char CgName[] = "CG";
    inline constexpr char GmresName[] = "DGMRES";

    /**
     * @brief Conjugate Gradient solver for symmetric positive definite matrices.
     */
    using CgOperator = EigenIterativeOperator<
        Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>, Eigen::Lower | Eigen::Upper, EigenPreconditionerAdapter>,
        CgName>;

    /**
     * @brief Dynamic GMRES solver for general unsymmetric matrices.
     */
    using GmresOperator = EigenIterativeOperator<
        Eigen::DGMRES<Eigen::SparseMatrix<Real>, EigenPreconditionerAdapter>,
        GmresName>;

    // =============================================================================
    // Direct Solver: Eigen SparseLU
    // =============================================================================

    class EigenSparseLUOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "SparseLU"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A)
                throw std::runtime_error("EigenSparseLUOperator: null matrix in setup");
            solver_.compute(A->eigen());
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("EigenSparseLUOperator: factorization failed");
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
    };

} // namespace mpfem

#endif // MPFEM_EIGEN_SOLVER_HPP

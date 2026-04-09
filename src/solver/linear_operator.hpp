#ifndef MPFEM_LINEAR_OPERATOR_HPP
#define MPFEM_LINEAR_OPERATOR_HPP

#include "core/types.hpp"
#include "solver/solver_config.hpp"
#include "core/sparse_matrix.hpp"
#include <memory>
#include <string>
#include <string_view>

namespace mpfem {

    // =============================================================================
    // Abstract Base Class
    // =============================================================================

    /**
     * @brief Unified abstract base class for all linear operators.
     *
     * Design principles:
     * - Two-stage lifecycle: setup(A) for heavy computation, apply(b,x) for pure iteration
     * - Does NOT own or destroy the input matrix (borrowed pointer only)
     * - Supports nested preconditioners via set_preconditioner()
     * - Configuration via virtual configure() method - no dynamic_cast needed
     */
    class LinearOperator {
    public:
        virtual ~LinearOperator() = default;

        /// Disable copy construction
        LinearOperator(const LinearOperator&) = delete;
        /// Disable copy assignment
        LinearOperator& operator=(const LinearOperator&) = delete;
        /// Enable move construction
        LinearOperator(LinearOperator&&) = default;
        /// Enable move assignment
        LinearOperator& operator=(LinearOperator&&) = default;

        /**
         * @brief Configure operator from configuration struct.
         *
         * Default implementation does nothing. Subclasses override to handle
         * type-specific parameters (MaxIterations, Tolerance, Shift, etc.).
         * This eliminates the need for dynamic_cast in factory code.
         *
         * @param config Configuration with type and parameters
         */
        virtual void configure(const LinearOperatorConfig& config)
        {
            (void)config;
        }

        /**
         * @brief Setup phase: heavy computation (factorization, preprocessing).
         * @param A Pointer to the system matrix (borrowed, NOT owned)
         */
        virtual void setup(const SparseMatrix* A) = 0;

        /**
         * @brief Apply phase: pure vector iteration/mapping.
         * @param b Right-hand side vector (input)
         * @param x Solution vector (output)
         *
         * Implements x = M^{-1} * b where M is this operator.
         */
        virtual void apply(const Vector& b, Vector& x) = 0;

        /**
         * @brief Set nested preconditioner for this operator.
         * @param pc Unique pointer to preconditioner (takes ownership)
         */
        void set_preconditioner(std::unique_ptr<LinearOperator> pc)
        {
            preconditioner_ = std::move(pc);
        }

        /// Get nested preconditioner (borrowed pointer, may be nullptr)
        LinearOperator* preconditioner() { return preconditioner_.get(); }
        const LinearOperator* preconditioner() const { return preconditioner_.get(); }

        /// Get operator name
        virtual std::string_view name() const = 0;

        /// Check if setup has been called
        bool is_setup() const { return is_setup_; }

        /// Reset operator state (clears cached data, keeps preconditioner)
        virtual void reset()
        {
            is_setup_ = false;
        }

        /// Get number of iterations from last apply (for iterative operators)
        virtual int iterations() const { return 0; }

        /// Get residual from last apply (for iterative operators)
        virtual Real residual() const { return Real {0}; }

    protected:
        LinearOperator() = default;

        /// Mark setup as complete (call after setup() succeeds)
        void mark_setup() { is_setup_ = true; }

        /// Get the borrowed matrix pointer
        const SparseMatrix* matrix() const { return matrix_; }

        /// Set the borrowed matrix pointer
        void set_matrix(const SparseMatrix* A) { matrix_ = A; }

    private:
        const SparseMatrix* matrix_ = nullptr; // Borrowed, not owned
        std::unique_ptr<LinearOperator> preconditioner_;
        bool is_setup_ = false;
    };

    // =============================================================================
    // Concrete Operator: DiagonalOperator (Jacobi preconditioner)
    // =============================================================================

    /**
     * @brief Diagonal (Jacobi) preconditioner.
     *
     * Stores the inverse of the diagonal and applies it via element-wise
     * vector multiplication. Very cheap but often effective for SPD problems.
     */
    class DiagonalOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "Diagonal"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("DiagonalOperator: null matrix in setup");
            }
            const Index n = A->rows();
            invDiag_.resize(n);
            for (Index i = 0; i < n; ++i) {
                const Real diag = A->coeff(i, i);
                if (std::abs(diag) < 1e-14) {
                    invDiag_(i) = 0.0;
                }
                else {
                    invDiag_(i) = 1.0 / diag;
                }
            }
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = invDiag_.cwiseProduct(b);
        }

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }

    private:
        Vector invDiag_;
    };

    // =============================================================================
    // Concrete Operator: IccOperator (Incomplete Cholesky)
    // =============================================================================

    /**
     * @brief Incomplete Cholesky preconditioner.
     *
     * Supports optional shift (diagonal perturbation) for stability with
     * nearly singular matrices. Only for symmetric positive definite matrices.
     */
    class IccOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "ICC"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("IccOperator: null matrix in setup");
            }
            solver_.setInitialShift(shift_);
            solver_.compute(A->eigen());
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solve(b);
        }

        void set_shift(Real shift) { shift_ = shift; }
        Real shift() const { return shift_; }

        void configure(const LinearOperatorConfig& config) override
        {
            if (auto it = config.parameters.find("Shift"); it != config.parameters.end()) {
                set_shift(it->second);
            }
        }

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }

    private:
        Real shift_ = 1e-14;
        Eigen::IncompleteCholesky<Real> solver_;
    };

    // =============================================================================
    // Concrete Operator: IluOperator (Incomplete LU)
    // =============================================================================

    /**
     * @brief Incomplete LU preconditioner without fill-in.
     *
     * General-purpose incomplete factorization for unsymmetric matrices.
     * Drop tolerance controls the fill-in level.
     */
    class IluOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "ILU"; }

        void setup(const SparseMatrix* A) override
        {
            if (!A) {
                throw std::runtime_error("IluOperator: null matrix in setup");
            }
            solver_.compute(A->eigen());
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            x = solver_.solve(b);
        }

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }

    private:
        Eigen::IncompleteLUT<Real> solver_;
    };

    // =============================================================================
    // Concrete Operator: AdditiveSchwarzOperator
    // =============================================================================

    /**
     * @brief Additive Schwarz domain decomposition preconditioner.
     *
     * Overlapping domain decomposition with local solves. The overlap level
     * and local solver type are configurable via nested preconditioner.
     */
    class AdditiveSchwarzOperator : public LinearOperator {
    public:
        std::string_view name() const override { return "AdditiveSchwarz"; }

        void setup(const SparseMatrix* A) override
        {
            set_matrix(A);
            mark_setup();
        }

        void apply(const Vector& b, Vector& x) override
        {
            // TODO: Implement domain decomposition with overlap
            x = b; // Fallback: identity
        }

        int iterations() const override { return 1; }
        Real residual() const override { return Real {0}; }
    };

} // namespace mpfem

#endif // MPFEM_LINEAR_OPERATOR_HPP
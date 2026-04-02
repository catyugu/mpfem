#ifndef MPFEM_PRECONDITIONER_HPP
#define MPFEM_PRECONDITIONER_HPP

#include "core/logger.hpp"
#include "core/types.hpp"
#include "solver/solver_config.hpp"
#include "solver/sparse_matrix.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <map>
#include <memory>
#include <string>

namespace mpfem {

    // =============================================================================
    // Abstract Preconditioner Base Class
    // =============================================================================

    /**
     * @brief Abstract base class for all preconditioners.
     *
     * Defines the interface for preconditioners used in iterative linear solvers.
     * Concrete implementations wrap Eigen's preconditioners.
     */
    class Preconditioner {
    public:
        virtual ~Preconditioner() = default;

        /// Disable copy construction
        Preconditioner(const Preconditioner&) = delete;
        /// Disable copy assignment
        Preconditioner& operator=(const Preconditioner&) = delete;
        /// Enable move construction
        Preconditioner(Preconditioner&&) = default;
        /// Enable move assignment
        Preconditioner& operator=(Preconditioner&&) = default;

        /**
         * @brief Apply the preconditioner: solve M*x = b for x.
         * @param A The system matrix (used for setup if needed).
         * @param x On input: initial guess; on output: preconditioned solution.
         * @param b The right-hand side vector.
         */
        virtual void apply(const SparseMatrix& A, Vector& x, const Vector& b) = 0;

        /**
         * @brief Set preconditioner parameters.
         * @param params Map of parameter name to value.
         */
        virtual void setParameters(const std::map<std::string, Real>& params) = 0;

        /**
         * @brief Get the name of this preconditioner.
         * @return Human-readable name.
         */
        virtual std::string name() const = 0;

    protected:
        Preconditioner() = default;
    };

    // =============================================================================
    // Diagonal (Jacobi) Preconditioner
    // =============================================================================

    /**
     * @brief Diagonal (Jacobi) preconditioner wrapping Eigen::DiagonalPreconditioner.
     *
     * Simple but effective preconditioner for SPD matrices.
     * Very cheap to compute and apply.
     */
    class DiagonalPreconditioner : public Preconditioner {
    public:
        DiagonalPreconditioner() = default;

        void apply(const SparseMatrix& A, Vector& x, const Vector& b) override;

        void setParameters(const std::map<std::string, Real>& params) override;

        std::string name() const override;

    private:
        Vector invDiag_;
        bool initialized_ = false;
    };

    // =============================================================================
    // Incomplete Cholesky Preconditioner (ICC)
    // =============================================================================

    /**
     * @brief Incomplete Cholesky preconditioner wrapping Eigen::IncompleteCholesky.
     *
     * Approximate Cholesky factorization for SPD matrices.
     * Supports a shift (damping) parameter for stability.
     */
    class ICCPreconditioner : public Preconditioner {
    public:
        ICCPreconditioner() : shift_(1e-14) { }

        void apply(const SparseMatrix& A, Vector& x, const Vector& b) override;

        void setParameters(const std::map<std::string, Real>& params) override;

        std::string name() const override;

        Real shift() const { return shift_; }

    private:
        Eigen::IncompleteCholesky<Real> preconditioner_;
        Real shift_;
        bool initialized_ = false;
        std::uint64_t fingerprint_ = 0;
    };

    // =============================================================================
    // Incomplete LU Preconditioner (ILU)
    // =============================================================================

    /**
     * @brief Incomplete LU preconditioner with fill-in wrapping Eigen::IncompleteLUT.
     *
     * General-purpose incomplete factorization for non-SPD matrices.
     * Supports fill level and drop tolerance parameters.
     */
    class ILUPreconditioner : public Preconditioner {
    public:
        ILUPreconditioner()
            : fillLevel_(10), dropTolerance_(1e-4) { }

        void apply(const SparseMatrix& A, Vector& x, const Vector& b) override;

        void setParameters(const std::map<std::string, Real>& params) override;

        std::string name() const override;

        int fillLevel() const { return fillLevel_; }
        Real dropTolerance() const { return dropTolerance_; }

    private:
        Eigen::IncompleteLUT<Real> preconditioner_;
        int fillLevel_;
        Real dropTolerance_;
        bool initialized_ = false;
        std::uint64_t fingerprint_ = 0;
    };

    // =============================================================================
    // Preconditioner Registry
    // =============================================================================

    /**
     * @brief Factory for creating preconditioner instances.
     *
     * Creates appropriate preconditioner based on PreconditionerType enum.
     * Returns nullptr for types that require hierarchical setup.
     */
    class PreconditionerRegistry {
    public:
        /**
         * @brief Create a preconditioner instance.
         * @param type The type of preconditioner to create.
         * @return unique_ptr to the preconditioner, or nullptr if type requires
         *         hierarchical setup (AdditiveSchwarz, AMG).
         */
        static std::unique_ptr<Preconditioner> create(PreconditionerType type)
        {
            switch (type) {
            case PreconditionerType::None:
                return nullptr;

            case PreconditionerType::Diagonal:
                return std::make_unique<DiagonalPreconditioner>();

            case PreconditionerType::ICC:
                return std::make_unique<ICCPreconditioner>();

            case PreconditionerType::ILU:
                return std::make_unique<ILUPreconditioner>();

            case PreconditionerType::AdditiveSchwarz:
                LOG_WARN << "AdditiveSchwarz preconditioner requires hierarchical setup; "
                         << "returning nullptr. Use HierarchicalSolver instead.";
                return nullptr;

            case PreconditionerType::AMG:
                LOG_WARN << "AMG preconditioner requires hierarchical setup; "
                         << "returning nullptr. Use HierarchicalSolver instead.";
                return nullptr;

            default:
                LOG_WARN << "Unknown preconditioner type, returning nullptr.";
                return nullptr;
            }
        }
    };

} // namespace mpfem

#endif // MPFEM_PRECONDITIONER_HPP

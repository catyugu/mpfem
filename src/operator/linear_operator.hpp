#ifndef MPFEM_LINEAR_OPERATOR_HPP
#define MPFEM_LINEAR_OPERATOR_HPP

#include "core/types.hpp"
#include "operator/parameter_list.hpp"
#include "operator/sparse_matrix.hpp"
#include <stdexcept>
#include <string>

namespace mpfem {

    /**
     * @brief Unified abstract base class for all linear operators.
     *
     * Memory management: This class uses a RAW POINTER OWNERSHIP MODEL.
     * Each operator OWNS its nested preconditioner (if any) via raw pointer.
     * The destructor deletes owned children.
     *
     * Design principles:
     * - Raw pointers throughout (no shared_ptr, no unique_ptr)
     * - Non-owning: setup() takes const SparseMatrix* (matrix outlives operator)
     * - Owning: inner_operator_ is deleted in destructor
     * - is_setup_ flag prevents apply() before setup()
     */
    class LinearOperator {
    public:
        virtual ~LinearOperator()
        {
            delete inner_operator_;
        }

        // =================================================================
        // 1. Parameter Configuration
        // =================================================================

        virtual void set_parameters(const ParameterList& params) = 0;

        // =================================================================
        // 2. Nested/Composition Interface
        // =================================================================

        /// Set nested operator (preconditioner). Takes ownership.
        virtual void set_preconditioner(LinearOperator* pc)
        {
            delete inner_operator_;
            inner_operator_ = pc;
        }

        /// Get nested operator (non-owning)
        virtual LinearOperator* get_preconditioner() const { return inner_operator_; }

        // =================================================================
        // 3. Build/Preprocess
        // =================================================================

        virtual void setup(const SparseMatrix* A) = 0;

        // =================================================================
        // 4. Core Execution
        // =================================================================

        virtual void apply(const Vector& b, Vector& x) = 0;

        virtual void apply_transpose(const Vector& b, Vector& x)
        {
            (void)b;
            (void)x;
            throw std::runtime_error("apply_transpose not implemented");
        }

        // =================================================================
        // State Queries
        // =================================================================

        bool is_setup() const { return is_setup_; }
        std::string get_name() const { return operator_name_; }
        virtual int num_iterations() const { return 0; }
        virtual Real residual() const { return 0.0; }

    protected:
        LinearOperator() = default;

        /// System matrix pointer (non-owning)
        const SparseMatrix* A_ = nullptr;

        /// Nested operator (OWNED - deleted in destructor)
        LinearOperator* inner_operator_ = nullptr;

        bool is_setup_ = false;
        std::string operator_name_ = "BaseOperator";
    };

} // namespace mpfem

#endif // MPFEM_LINEAR_OPERATOR_HPP
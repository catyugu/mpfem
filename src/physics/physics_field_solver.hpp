#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "assembly/assembler.hpp"
#include "core/logger.hpp"
#include "fe/coefficient.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "operator/linear_operator.hpp"
#include "physics/field_values.hpp"
#include <memory>

namespace mpfem {

    class PhysicsFieldSolver {
    public:
        virtual ~PhysicsFieldSolver() = default;

        virtual std::string fieldName() const = 0;
        virtual FieldId fieldId() const = 0;
        virtual void assemble() = 0;

        bool solve()
        {
            if (!solver_ || !matAsm_ || !vecAsm_ || !fieldValues_)
                return false;

            // Only setup operator if matrix was rebuilt (tracked by derived classes)
            if (matrix_needs_update_) {
                solver_->setup(&matAsm_->matrix());
                matrix_needs_update_ = false;
            }

            // Apply solver: x = solver^{-1} * b (x used as initial guess for iterative solvers)
            Vector& x = field().values();
            const Vector& b = vecAsm_->vector();
            solver_->apply(b, x);

            field().markUpdated();
            LOG_INFO << fieldName() << " solved (iterations: " << solver_->num_iterations()
                     << ", residual: " << solver_->residual() << ")";
            return true;
        }

        const GridFunction& field() const
        {
            MPFEM_ASSERT(fieldValues_ != nullptr, "FieldValues not set");
            return fieldValues_->current(fieldId());
        }
        GridFunction& field()
        {
            MPFEM_ASSERT(fieldValues_ != nullptr, "FieldValues not set");
            return fieldValues_->current(fieldId());
        }

        const FESpace& feSpace() const { return *fes_; }
        Index numDofs() const { return fes_ ? fes_->numDofs() : 0; }
        const Mesh& mesh() const { return *mesh_; }

        void setSolverOperator(std::unique_ptr<LinearOperator> op) { solver_ = std::move(op); }

        int iterations() const { return solver_ ? solver_->num_iterations() : 0; }
        Real residual() const { return solver_ ? solver_->residual() : 0.0; }

    protected:
        void clearAssemblers()
        {
            if (matAsm_) {
                matAsm_->clear();
                matAsm_->clearIntegrators();
            }
            if (vecAsm_) {
                vecAsm_->clear();
                vecAsm_->clearIntegrators();
            }
        }

        /// Flag indicating matrix needs re-setup (set by derived class after assemble()
        bool matrix_needs_update_ = true;

        int order_ = 1;

        const Mesh* mesh_ = nullptr;
        FieldValues* fieldValues_ = nullptr;
        std::unique_ptr<FESpace> fes_;
        std::unique_ptr<BilinearFormAssembler> matAsm_;
        std::unique_ptr<LinearFormAssembler> vecAsm_;
        std::unique_ptr<LinearOperator> solver_;
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_FIELD_SOLVER_HPP

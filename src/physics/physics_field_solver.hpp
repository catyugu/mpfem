#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "assembly/assembler.hpp"
#include "core/logger.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "physics/field_values.hpp"
#include "solver/linear_operator.hpp"
#include "solver/solver_config.hpp"
#include <memory>

namespace mpfem {

    class PhysicsFieldSolver {
    public:
        virtual ~PhysicsFieldSolver() = default;

        virtual std::string fieldName() const = 0;
        virtual FieldId fieldId() const = 0;
        virtual void assemble() = 0;

        void solve()
        {
            MPFEM_ASSERT(solver_ != nullptr, "Solver not configured - call setSolverConfig() first");
            MPFEM_ASSERT(matAsm_ != nullptr, "Matrix assembler not configured");
            MPFEM_ASSERT(vecAsm_ != nullptr, "Vector assembler not configured");
            MPFEM_ASSERT(fieldValues_ != nullptr, "FieldValues not set");

            solver_->setup(&matAsm_->matrix());
            solver_->apply(vecAsm_->vector(), field().values());

            field().markUpdated();
            LOG_INFO << fieldName() << " solver iterations: " << solver_->iterations()
                     << ", residual: " << solver_->residual();
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

        void setSolverConfig(std::unique_ptr<LinearOperatorConfig> config) { solverConfig_ = std::move(config); }

        int iterations() const { return solver_->iterations(); }
        Real residual() const { return solver_->residual(); }

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

        int order_ = 1;
        std::unique_ptr<LinearOperatorConfig> solverConfig_;

        const Mesh* mesh_ = nullptr;
        FieldValues* fieldValues_ = nullptr;
        std::unique_ptr<FESpace> fes_;
        std::unique_ptr<BilinearFormAssembler> matAsm_;
        std::unique_ptr<LinearFormAssembler> vecAsm_;
        std::unique_ptr<LinearOperator> solver_;
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_FIELD_SOLVER_HPP
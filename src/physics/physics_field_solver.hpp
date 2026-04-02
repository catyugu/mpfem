#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "assembly/assembler.hpp"
#include "core/logger.hpp"
#include "fe/coefficient.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "physics/field_values.hpp"
#include "solver/linear_solver.hpp"
#include "solver/solver_config.hpp"
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
            bool ok = solver_->solve(matAsm_->matrix(), field().values(), vecAsm_->vector());
            if (ok) {
                field().markUpdated();
                LOG_INFO << fieldName() << " solver converged!";
            }
            return ok;
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

        void setSolverConfig(SolverConfig config) { solverConfig_ = std::move(config); }

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
        SolverConfig solverConfig_;

        const Mesh* mesh_ = nullptr;
        FieldValues* fieldValues_ = nullptr;
        std::unique_ptr<FESpace> fes_;
        std::unique_ptr<BilinearFormAssembler> matAsm_;
        std::unique_ptr<LinearFormAssembler> vecAsm_;
        std::unique_ptr<LinearSolver> solver_;
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_FIELD_SOLVER_HPP

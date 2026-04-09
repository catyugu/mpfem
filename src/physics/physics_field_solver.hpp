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

        // Matrix building interface - Problem class orchestrates assembly
        virtual void buildStiffnessMatrix(SparseMatrix& K) = 0;
        virtual void buildMassMatrix(SparseMatrix& M) { M.resize(0, 0); } // Default: no mass
        virtual void buildRHS(Vector& F) = 0;
        virtual void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution) = 0;

        // Default linear system solve implementation
        virtual bool solveLinearSystem(SparseMatrix& A, Vector& x, const Vector& b)
        {
            A.makeCompressed();
            solver_->setup(&A);
            solver_->apply(b, x);
            field().markUpdated();
            return true;
        }

        const GridFunction& field() const
        {
            MPFEM_ASSERT(fieldValues_ != nullptr, "FieldValues not set");
            return fieldValues_->current(fieldName());
        }
        GridFunction& field()
        {
            MPFEM_ASSERT(fieldValues_ != nullptr, "FieldValues not set");
            return fieldValues_->current(fieldName());
        }

        const FESpace& feSpace() const { return *fes_; }
        Index numDofs() const { return fes_ ? fes_->numDofs() : 0; }
        const Mesh& mesh() const { return *mesh_; }

        void setSolverConfig(std::unique_ptr<LinearOperatorConfig> config) { solverConfig_ = std::move(config); }

        int iterations() const { return solver_->iterations(); }
        Real residual() const { return solver_->residual(); }

    protected:
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
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

        // Public solve interface (Template Method pattern - final, not overridable)
        bool solveSteady()
        {

            buildStiffnessMatrix(K_uneliminated_);

            // Build RHS
            buildRHS(F_);

            // Prepare eliminated matrix
            K_eliminated_ = K_uneliminated_;

            // Apply essential boundary conditions
            applyEssentialBCs(K_eliminated_, F_, field().values());


            K_eliminated_.makeCompressed();
            solver_->setup(&K_eliminated_);

            // Solve
            solver_->apply(F_, field().values());
            field().markUpdated();
            return true;
        }

        bool solveTransient(Real dt, const Vector& historyCombo)
        {

            buildStiffnessMatrix(K_uneliminated_);
            buildMassMatrix(M_);     

            bool M_is_empty = (M_.rows() == 0 || M_.cols() == 0);
            if (M_is_empty) {
                A_uneliminated_ = dt * K_uneliminated_;
            }
            else {
                A_uneliminated_ = M_ + (dt * K_uneliminated_);
            }
            previous_dt_ = dt;

            // Build RHS
            buildRHS(F_);

            // Compute transient RHS = M*historyCombo + dt*F (or dt*F if M is empty)
            Vector transient_rhs;
            if (M_is_empty) {
                transient_rhs = dt * F_;
            }
            else {
                transient_rhs = M_ * historyCombo + (dt * F_);
            }

            // Prepare eliminated matrix
            A_eliminated_ = A_uneliminated_;

            // Apply essential boundary conditions
            applyEssentialBCs(A_eliminated_, transient_rhs, field().values());

            A_eliminated_.makeCompressed();
            solver_->setup(&A_eliminated_);

            // Solve
            solver_->apply(transient_rhs, field().values());
            field().markUpdated();
            return true;
        }

        // Mark matrix as changed to force rebuild on next solve
        void markMatrixChanged() { systemMatrixNeedsRebuild_ = true; }

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

        // Template Method interfaces - Problem class orchestrates assembly
        virtual void buildStiffnessMatrix(SparseMatrix& K) = 0;
        virtual void buildMassMatrix(SparseMatrix& M) { M.resize(0, 0); } // Default: no mass
        virtual void buildRHS(Vector& F) = 0;
        virtual void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution) = 0;

        // Cached matrices
        SparseMatrix K_uneliminated_, K_eliminated_;
        SparseMatrix M_;
        SparseMatrix A_uneliminated_, A_eliminated_;
        Vector F_;
        Real previous_dt_ = -1.0;
        bool systemMatrixNeedsRebuild_ = true;
        bool solverNeedsSetup_ = true;
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_FIELD_SOLVER_HPP
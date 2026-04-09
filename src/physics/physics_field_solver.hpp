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
            std::uint64_t currentMatRev = getMatrixRevision();
            std::uint64_t currentRhsRev = getRhsRevision();
            std::uint64_t currentBcRev = getBcRevision();

            bool matOrBcChanged = (currentMatRev != matrixRevision_) || (currentBcRev != bcRevision_) || isFirstIteration;
            bool rhsChanged = (currentRhsRev != rhsRevision_) || isFirstIteration;

            if (matOrBcChanged) {
                buildStiffnessMatrix(K_uneliminated_);
                matrixRevision_ = currentMatRev;
            }

            if (matOrBcChanged || rhsChanged) {
                buildRHS(F_);
                rhsRevision_ = currentRhsRev;
            }

            if (matOrBcChanged) {
                // Prepare eliminated matrix
                K_eliminated_ = K_uneliminated_;

                // Apply essential boundary conditions - update matrix and RHS
                applyEssentialBCs(K_eliminated_, F_, field().values(), true);

                K_eliminated_.makeCompressed();
                solver_->setup(&K_eliminated_);
                bcRevision_ = currentBcRev;
            }
            else if (rhsChanged) {
                // Fast path: only update RHS for changed source terms
                applyEssentialBCs(K_eliminated_, F_, field().values(), false);
            }

            // Solve
            solver_->apply(F_, field().values());
            field().markUpdated();
            isFirstIteration = false;
            
            return true;
        }

        bool solveTransient(Real dt, const Vector& historyCombo)
        {
            std::uint64_t currentMatRev = getMatrixRevision();
            std::uint64_t currentMassRev = getMassRevision();
            std::uint64_t currentRhsRev = getRhsRevision();
            std::uint64_t currentBcRev = getBcRevision();

            bool dtChanged = (dt != previous_dt_);
            bool operatorChanged = (currentMatRev != matrixRevision_) || (currentMassRev != massRevision_) || dtChanged || isFirstIteration ;
            bool rhsChanged = (currentRhsRev != rhsRevision_) || isFirstIteration;

            if (operatorChanged) {
                buildStiffnessMatrix(K_uneliminated_);
                buildMassMatrix(M_);
                matrixRevision_ = currentMatRev;
                massRevision_ = currentMassRev;
            }

            if (operatorChanged || rhsChanged) {
                buildRHS(F_);
                rhsRevision_ = currentRhsRev;
            }

            // Compute transient RHS = M*historyCombo + dt*F (or dt*F if M is empty)
            bool M_is_empty = (M_.rows() == 0 || M_.cols() == 0);
            Vector transient_rhs;
            if (M_is_empty) {
                transient_rhs = dt * F_;
            }
            else {
                transient_rhs = M_ * historyCombo + (dt * F_);
            }

            if (operatorChanged) {
                bool M_is_empty_check = (M_.rows() == 0 || M_.cols() == 0);
                if (M_is_empty_check) {
                    A_uneliminated_ = dt * K_uneliminated_;
                }
                else {
                    A_uneliminated_ = M_ + (dt * K_uneliminated_);
                }
                previous_dt_ = dt;

                // Prepare eliminated matrix and apply BCs with full matrix update
                A_eliminated_ = A_uneliminated_;
                applyEssentialBCs(A_eliminated_, transient_rhs, field().values(), true);
                A_eliminated_.makeCompressed();
                solver_->setup(&A_eliminated_);
                bcRevision_ = currentBcRev;
            }
            else if (rhsChanged || dtChanged) {
                // Fast path: matrix unchanged, only update RHS
                applyEssentialBCs(A_eliminated_, transient_rhs, field().values(), false);
            }

            // Solve
            solver_->apply(transient_rhs, field().values());
            field().markUpdated();
            isFirstIteration = false;
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

        // Template Method interfaces - Problem class orchestrates assembly
        virtual void buildStiffnessMatrix(SparseMatrix& K) = 0;
        virtual void buildMassMatrix(SparseMatrix& M) { M.resize(0, 0); } // Default: no mass
        virtual void buildRHS(Vector& F) = 0;
        virtual void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix) = 0;

        // Revision tracking for smart matrix assembly skipping
        virtual std::uint64_t getMatrixRevision() const { return matrixRevision_; }
        virtual std::uint64_t getMassRevision() const { return massRevision_; }
        virtual std::uint64_t getRhsRevision() const { return rhsRevision_; }
        virtual std::uint64_t getBcRevision() const { return bcRevision_; }

        // Cached matrices
        SparseMatrix K_uneliminated_, K_eliminated_;
        SparseMatrix M_;
        SparseMatrix A_uneliminated_, A_eliminated_;
        Vector F_;
        Real previous_dt_ = -1.0;

        bool isFirstIteration = true; // Flag to track first iteration for mandatory initial assembly

        // Revision tracking state
        std::uint64_t matrixRevision_ = 0;
        std::uint64_t massRevision_ = 0;
        std::uint64_t rhsRevision_ = 0;
        std::uint64_t bcRevision_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_FIELD_SOLVER_HPP

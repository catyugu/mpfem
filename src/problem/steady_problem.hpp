#ifndef MPFEM_STEADY_PROBLEM_HPP
#define MPFEM_STEADY_PROBLEM_HPP

#include "core/logger.hpp"
#include "core/types.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/field_values.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "solver/sparse_matrix.hpp"

namespace mpfem {

    struct SteadyResult {
        bool converged = false;
        int iterations = 0;
        Real residual = 0.0;
        FieldValues fields; ///< Final field values after solve
    };

    class SteadyProblem : public Problem {
    public:
        SteadyResult solve()
        {
            ScopedTimer timer("Coupling solve");
            SteadyResult result;

            if (!isCoupled()) {
                if (hasElectrostatics()) {
                    SparseMatrix K;
                    Vector F;
                    electrostatics->buildStiffnessMatrix(K);
                    electrostatics->buildRHS(F);
                    electrostatics->applyBoundaryConditions(K, F, electrostatics->field().values());
                    electrostatics->solveLinearSystem(K, electrostatics->field().values(), F);
                }
                result.fields = fieldValues;
                return result;
            }

            // Hoist physics checks outside loop to avoid per-iteration branching
            const bool hasElectrostatics = this->hasElectrostatics();
            const bool hasHeatTransfer = this->hasHeatTransfer();

            for (int i = 0; i < couplingMaxIter; ++i) {
                if (hasElectrostatics) {
                    SparseMatrix K;
                    Vector F;
                    electrostatics->buildStiffnessMatrix(K);
                    electrostatics->buildRHS(F);
                    electrostatics->applyBoundaryConditions(K, F, electrostatics->field().values());
                    electrostatics->solveLinearSystem(K, electrostatics->field().values(), F);
                }
                if (hasHeatTransfer) {
                    SparseMatrix K;
                    Vector F;
                    heatTransfer->buildStiffnessMatrix(K);
                    heatTransfer->buildRHS(F);
                    heatTransfer->applyBoundaryConditions(K, F, heatTransfer->field().values());
                    heatTransfer->solveLinearSystem(K, heatTransfer->field().values(), F);
                }

                Real err = computeCouplingError();
                result.iterations = i + 1;
                result.residual = err;
                LOG_INFO << "Coupling iteration " << (i + 1) << ", residual = " << err;
                if (err < couplingTol) {
                    result.converged = true;
                    break;
                }
            }

            if (hasStructural()) {
                SparseMatrix K;
                Vector F;
                structural->buildStiffnessMatrix(K);
                structural->buildRHS(F);
                structural->applyBoundaryConditions(K, F, structural->field().values());
                structural->solveLinearSystem(K, structural->field().values(), F);
            }
            result.fields = fieldValues;
            return result;
        }

    private:
        Vector prevT_;

        Real computeCouplingError()
        {
            if (!heatTransfer)
                return 0.0;
            const auto& T = heatTransfer->field().values();
            if (prevT_.size() == 0) {
                prevT_ = T;
                return 1.0;
            }
            Real diff = (T - prevT_).norm();
            prevT_ = T;
            return diff / (T.norm() + 1e-15);
        }
    };

} // namespace mpfem

#endif // MPFEM_STEADY_PROBLEM_HPP
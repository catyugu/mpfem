#ifndef MPFEM_TRANSIENT_PROBLEM_HPP
#define MPFEM_TRANSIENT_PROBLEM_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "problem.hpp"
#include "time/time_scheme.hpp"
#include <memory>

namespace mpfem {

    /**
     * @brief Result of transient solve
     */
    struct TransientResult {
        bool converged = false;
        int timeSteps = 0;
        Real finalTime = 0.0;

        std::vector<Real> times; ///< Time at each snapshot
        std::vector<FieldValues> snapshots; ///< Field values at each time step

        void addSnapshot(Real time, const FieldValues& fields)
        {
            times.push_back(time);
            snapshots.emplace_back();
            snapshots.back() = fields; // Copy
        }

        int numTimeSteps() const { return static_cast<int>(times.size()); }
    };

    class TransientProblem : public Problem {
    public:
        bool isTransient() const override { return true; }

        Real startTime = 0.0;
        Real endTime = 1.0;
        Real timeStep = 0.01;
        Real currentTime = 0.0;
        int currentStep = 0;

        TimeScheme scheme = TimeScheme::BDF1;

        void initializeTransient(int historyDepth = 2)
        {
            fieldValues.setMaxHistorySteps(historyDepth);
        }

        const GridFunction& history(std::string_view id, int stepsBack = 1) const
        {
            return fieldValues.history(id, stepsBack);
        }

        GridFunction& history(std::string_view id, int stepsBack = 1)
        {
            return fieldValues.history(id, stepsBack);
        }

        /// @brief Solve steady-state initialization at t=0 before time stepping
        void initializeSteadyState();

        /**
         * @brief Solve the transient problem with time stepping and Picard coupling
         *
         * Outer loop: time stepping
         * Inner loop: Picard coupling iteration
         *   - Electrostatics: quasi-static, temperature-dependent conductivity
         *   - Heat: transient (handled by time integrator)
         *   - Structural: quasi-static, thermal stress
         */
        TransientResult solve();
    };

} // namespace mpfem

#endif // MPFEM_TRANSIENT_PROBLEM_HPP

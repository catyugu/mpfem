#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "expr/variable_graph.hpp"
#include "physics_field_solver.hpp"
#include <set>
#include <vector>

namespace mpfem {

    /**
     * @brief Heat transfer solver
     *
     * Solves: -div(k * grad T) = Q
     * k is a 3x3 thermal conductivity tensor (matrix coefficient)
     */
    class HeatTransferSolver : public PhysicsFieldSolver {
    public:
        HeatTransferSolver() = default;
        explicit HeatTransferSolver(int order) { order_ = order; }
        std::string fieldName() const override { return "HeatTransfer"; }

        bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialTemperature = 293.15);

        // Material bindings
        void setThermalConductivity(const std::set<int>& domains, const VariableNode* k);

        void setHeatSource(const std::set<int>& domains, const VariableNode* Q);
        void setMassProperties(const std::set<int>& domains, const VariableNode* rhoCp);

        // Mass matrix for transient terms: M = ∫ ρCp φᵢ φⱼ dΩ
        void assembleMassMatrix();
        const SparseMatrix& massMatrix() const { return massMatrix_; }
        bool massMatrixAssembled() const { return massMatrixAssembled_; }

        // Boundary conditions
        void addTemperatureBC(const std::set<int>& boundaryIds, const VariableNode* temperature);
        void addConvectionBC(const std::set<int>& boundaryIds, const VariableNode* h, const VariableNode* Tinf);
        void clearBoundaryConditions()
        {
            temperatureBindings_.clear();
            convectionBindings_.clear();
        }

        void assemble() override;

        /**
         * @brief Solve a custom linear system Ax = b with applied boundary conditions
         *
         * This is useful for time integrators like BDF1 that need to solve
         * a modified system (e.g., M + dt*K) rather than the standard K.
         *
         * @param A System matrix
         * @param x Solution vector (output)
         * @param b Right-hand side vector
         * @return true if solved successfully
         */
        bool solveLinearSystem(SparseMatrix& A, Vector& x, const Vector& b);

        /**
         * @brief Get stiffness matrix before BC application
         *
         * This is needed for transient time integrators like BDF1 that need
         * to form the combined system (M + dt*K) before BCs are applied.
         *
         * @return Stiffness matrix K (without BC modifications)
         */
        const SparseMatrix& stiffnessMatrixBeforeBC() const { return stiffnessMatrixBeforeBC_; }

        /**
         * @brief Get RHS vector before BC application
         *
         * This is needed for transient time integrators like BDF1 that need
         * to form the combined RHS (M*T_prev + dt*Q) before BCs are applied.
         *
         * @return RHS vector Q (without BC modifications)
         */
        const Vector& rhsBeforeBC() const { return rhsBeforeBC_; }

    private:
        struct TemperatureBinding {
            std::set<int> boundaryIds;
            const VariableNode* temperature = nullptr;
        };

        struct ConvectionBinding {
            std::set<int> boundaryIds;
            const VariableNode* h = nullptr;
            const VariableNode* Tinf = nullptr;
        };

        struct ConductivityBinding {
            std::set<int> domains;
            const VariableNode* conductivity = nullptr;
        };
        struct HeatSourceBinding {
            std::set<int> domains;
            const VariableNode* source = nullptr;
        };
        struct MassBinding {
            std::set<int> domains;
            const VariableNode* thermalMass = nullptr;
        };

        std::vector<ConductivityBinding> conductivityBindings_;
        std::vector<HeatSourceBinding> heatSourceBindings_;
        std::vector<MassBinding> massBindings_;
        std::vector<TemperatureBinding> temperatureBindings_;
        std::vector<ConvectionBinding> convectionBindings_;
        SparseMatrix massMatrix_;
        bool massMatrixAssembled_ = false;

        /// @brief Stiffness matrix before BC application (for transient time integrators)
        SparseMatrix stiffnessMatrixBeforeBC_;

        /// @brief RHS vector before BC application (for transient time integrators)
        Vector rhsBeforeBC_;

        // Reusable buffers for transient linear systems after BC application.
        SparseMatrix systemMatrix_;
        Vector systemRhs_;
    };

} // namespace mpfem

#endif // MPFEM_HEAT_TRANSFER_SOLVER_HPP
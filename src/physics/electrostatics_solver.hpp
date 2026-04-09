#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "expr/variable_graph.hpp"
#include "physics_field_solver.hpp"
#include <set>
#include <vector>

namespace mpfem {

    /**
     * @brief Electrostatics solver
     *
     * Solves: -div(sigma * grad V) = 0
     * sigma is a 3x3 conductivity tensor (matrix coefficient)
     */
    class ElectrostaticsSolver : public PhysicsFieldSolver {
    public:
        ElectrostaticsSolver() = default;
        explicit ElectrostaticsSolver(int order) { order_ = order; }

        std::string fieldName() const override { return "V"; }
        int VDim() const override { return 1; } // Scalar field

        // Material bindings
        void setElectricalConductivity(const std::set<int>& domains, const VariableNode* sigma);

        // Boundary conditions
        void addVoltageBC(const std::set<int>& boundaryIds, const VariableNode* voltage);
        void clearBoundaryConditions() { voltageBindings_.clear(); }

    protected:
        void buildStiffnessMatrix(SparseMatrix& K) override;
        void buildMassMatrix(SparseMatrix& M) override { M.resize(0, 0); }
        void buildRHS(Vector& F) override;
        void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix) override;

        std::uint64_t getMatrixRevision() const override;
        std::uint64_t getMassRevision() const override { return 0; }
        std::uint64_t getRhsRevision() const override;
        std::uint64_t getBcRevision() const override;

    private:
        struct ConductivityBinding {
            std::set<int> domains;
            const VariableNode* sigma = nullptr;
        };

        struct VoltageBinding {
            std::set<int> boundaryIds;
            const VariableNode* voltage = nullptr;
        };

        std::vector<ConductivityBinding> conductivityBindings_;
        std::vector<VoltageBinding> voltageBindings_;
    };

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
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
        std::string fieldName() const override { return "T"; }

        bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, Real initialTemperature = 293.15);

        // Material bindings
        void setThermalConductivity(const std::set<int>& domains, const VariableNode* k);

        void setHeatSource(const std::set<int>& domains, const VariableNode* Q);
        void setMassProperties(const std::set<int>& domains, const VariableNode* rhoCp);

        // Boundary conditions
        void addTemperatureBC(const std::set<int>& boundaryIds, const VariableNode* temperature);
        void addConvectionBC(const std::set<int>& boundaryIds, const VariableNode* h, const VariableNode* Tinf);
        void clearBoundaryConditions()
        {
            temperatureBindings_.clear();
            convectionBindings_.clear();
        }

    protected:
        void buildStiffnessMatrix(SparseMatrix& K) override;
        void buildMassMatrix(SparseMatrix& M) override;
        void buildRHS(Vector& F) override;
        void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix) override;

        std::uint64_t getMatrixRevision() const override;
        std::uint64_t getMassRevision() const override;
        std::uint64_t getRhsRevision() const override;
        std::uint64_t getBcRevision() const override;

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
    };

} // namespace mpfem

#endif // MPFEM_HEAT_TRANSFER_SOLVER_HPP
#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "expr/variable_graph.hpp"
#include "physics_field_solver.hpp"
#include <set>
#include <vector>

namespace mpfem {

    /**
     * @brief Structural mechanics solver (linear elasticity)
     *
     * Solves: -div(sigma) = 0
     * where sigma = C : epsilon
     */
    class StructuralSolver : public PhysicsFieldSolver {
    public:
        StructuralSolver() = default;
        explicit StructuralSolver(int order) { order_ = order; }

        std::string fieldName() const override { return "u"; }

        bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, Real initialDisplacement = 0.0);

        // Material bindings
        void addElasticity(const std::set<int>& domains, const VariableNode* E, const VariableNode* nu);

        // Boundary conditions
        void addFixedDisplacementBC(const std::set<int>& boundaryIds, const VariableNode* displacement);
        void clearBoundaryConditions() { displacementBindings_.clear(); }

        // Generic stress load term assembled as ∫ sigma : epsilon(v) dΩ
        void setStrainLoad(const std::set<int>& domains, const VariableNode* stress);
        bool hasStrainLoad() const { return !strainLoadBindings_.empty(); }

    protected:
        void buildStiffnessMatrix(SparseMatrix& K) override;
        void buildMassMatrix(SparseMatrix& M) override { M.resize(0, 0); }
        void buildRHS(Vector& F) override;
        void applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix) override;

        std::uint64_t getMatrixRevision() const override;
        std::uint64_t getMassRevision() const override { return 0; }
        std::uint64_t getRhsRevision() const override;
        std::uint64_t getBcRevision() const override { return 0; }

    private:
        struct ElasticityBinding {
            std::set<int> domains;
            const VariableNode* E = nullptr;
            const VariableNode* nu = nullptr;
        };
        struct StrainLoadBinding {
            std::set<int> domains;
            const VariableNode* stress = nullptr;
        };

        struct DisplacementBinding {
            std::set<int> boundaryIds;
            const VariableNode* displacement = nullptr;
        };

        std::vector<ElasticityBinding> elasticityBindings_;
        std::vector<StrainLoadBinding> strainLoadBindings_;
        std::vector<DisplacementBinding> displacementBindings_;
    };

} // namespace mpfem

#endif // MPFEM_STRUCTURAL_SOLVER_HPP
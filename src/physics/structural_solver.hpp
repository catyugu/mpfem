#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "assembly_change_tracker.hpp"
#include "fe/coefficient.hpp"
#include "physics_field_solver.hpp"
#include <cstdint>
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

        std::string fieldName() const override { return "Structural"; }
        FieldId fieldId() const override { return FieldId::Displacement; }

        bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialDisplacement = 0.0);

        // Material bindings
        void addElasticity(const std::set<int>& domains, const Coefficient* E, const Coefficient* nu);

        // Boundary conditions
        void addFixedDisplacementBC(const std::set<int>& boundaryIds, const VectorCoefficient* displacement);
        void clearBoundaryConditions() { displacementBindings_.clear(); }

        // Generic stress load term assembled as ∫ sigma : epsilon(v) dΩ
        void setStrainLoad(const std::set<int>& domains, const MatrixCoefficient* stress);
        bool hasStrainLoad() const { return !strainLoadBindings_.empty(); }

        void assemble() override;

    private:
        struct ElasticityBinding {
            std::set<int> domains;
            const Coefficient* E = nullptr;
            const Coefficient* nu = nullptr;

            std::uint64_t stateTag() const
            {
                return combineTag(
                    stateTagOf(domains),
                    combineTag(stateTagOf(E), stateTagOf(nu)));
            }
        };
        struct StrainLoadBinding {
            std::set<int> domains;
            const MatrixCoefficient* stress = nullptr;

            std::uint64_t stateTag() const
            {
                return combineTag(stateTagOf(domains), stateTagOf(stress));
            }
        };

        struct DisplacementBinding {
            std::set<int> boundaryIds;
            const VectorCoefficient* displacement = nullptr;

            std::uint64_t stateTag() const
            {
                return combineTag(stateTagOf(boundaryIds), stateTagOf(displacement));
            }
        };

        std::vector<ElasticityBinding> elasticityBindings_;
        std::vector<StrainLoadBinding> strainLoadBindings_;
        std::vector<DisplacementBinding> displacementBindings_;

        SparseMatrix stiffnessMatrixBeforeBC_;
        Vector rhsBeforeBC_;
        AssemblyTagCache stiffnessAssemblyState_;
        AssemblyTagCache loadAssemblyState_;
        AssemblyTagCache bcAssemblyState_;
    };

} // namespace mpfem

#endif // MPFEM_STRUCTURAL_SOLVER_HPP
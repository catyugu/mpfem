#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly_change_tracker.hpp"
#include "fe/coefficient.hpp"
#include <cstdint>
#include <map>
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
    
    std::string fieldName() const override { return "Electrostatics"; }
    FieldId fieldId() const override { return FieldId::ElectricPotential; }
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialPotential = 0.0);
    
    // Material bindings
    void setElectricalConductivity(const std::set<int>& domains, const MatrixCoefficient* sigma);
    
    // Boundary conditions
    void addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage);
    void clearBoundaryConditions() { voltageBCs_.clear(); }
    
    void assemble() override;

private:
    struct ConductivityBinding {
        std::set<int> domains;
        const MatrixCoefficient* sigma = nullptr;

        std::uint64_t stateTag() const {
            return combineTag(stateTagOf(domains), stateTagOf(sigma));
        }
    };

    std::vector<ConductivityBinding> conductivityBindings_;
    std::map<int, const Coefficient*> voltageBCs_;
    AssemblyTagCache stiffnessAssemblyState_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
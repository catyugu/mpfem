#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <vector>
#include <set>

namespace mpfem {

/**
 * @brief Electrostatics solver
 * 
 * Solves: -div(sigma * grad V) = 0
 * 
 * Design principles:
 * - Single-field solver does not contain coupling logic
 * - Conductivity coefficient supports domain selection, each domain can use different coefficients
 * - Boundary conditions use Coefficient instead of direct values
 * - Field values are owned by FieldValues, solver holds reference
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    /// Initialize solver (creates FESpace, registers field with FieldValues)
    /// @param mesh The mesh
    /// @param fieldValues Field value manager (solver registers its field here)
    bool initialize(const Mesh& mesh, FieldValues& fieldValues);
    
    // =========================================================================
    // Material coefficient interfaces (support domain selection)
    // =========================================================================
    
    /// Set conductivity coefficient for specified domains (non-owning pointer, overwrites existing)
    void setConductivity(const std::set<int>& domains, const Coefficient* sigma);
    
    /// Set conductivity coefficient for all domains (non-owning pointer, for coupling coefficient)
    void setConductivity(const Coefficient* sigma) {
        conductivity_.setAll(sigma);
    }
    
    /// Get conductivity coefficient
    const DomainMappedScalarCoefficient& conductivity() const { return conductivity_; }
    
    // =========================================================================
    // Boundary condition interfaces (use Coefficient)
    // =========================================================================
    
    /// Add voltage boundary condition (batch setting, use Coefficient)
    void addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage);
    
    /// Clear boundary conditions
    void clearBoundaryConditions() { voltageBCs_.clear(); }
    
    // =========================================================================
    // Solve interface
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return fieldValues_->current(FieldId::ElectricPotential); }
    GridFunction& field() override { return fieldValues_->current(FieldId::ElectricPotential); }

private:
    FieldValues* fieldValues_ = nullptr;  ///< Non-owning reference to field manager
    DomainMappedScalarCoefficient conductivity_;
    std::map<int, const Coefficient*> voltageBCs_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
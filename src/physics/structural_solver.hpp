#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <set>

namespace mpfem {

/**
 * @brief Structural mechanics solver (linear elasticity)
 * 
 * Solves: -div(sigma) = 0
 * where sigma = C : epsilon
 * 
 * Design principles:
 * - Single-field solver does not contain coupling logic
 * - Material coefficients support domain selection
 * - Boundary conditions use Coefficient
 * - Field values are owned by FieldValues, solver holds reference
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// Initialize solver (creates FESpace, registers field with FieldValues)
    /// @param mesh The mesh
    /// @param fieldValues Field value manager (solver registers its field here)
    bool initialize(const Mesh& mesh, FieldValues& fieldValues);
    
    // =========================================================================
    // Material coefficient interfaces (support domain selection)
    // =========================================================================
    
    /// Set Young's modulus coefficient for specified domains
    void setYoungModulus(const std::set<int>& domains, const Coefficient* E);
    
    /// Set Young's modulus coefficient for all domains
    void setYoungModulus(const Coefficient* E) {
        youngModulus_.setAll(E);
    }
    
    /// Set Poisson's ratio coefficient for specified domains
    void setPoissonRatio(const std::set<int>& domains, const Coefficient* nu);
    
    /// Set Poisson's ratio coefficient for all domains
    void setPoissonRatio(const Coefficient* nu) {
        poissonRatio_.setAll(nu);
    }
    
    /// Get Young's modulus coefficient
    const DomainMappedScalarCoefficient& youngModulus() const { return youngModulus_; }
    
    /// Get Poisson's ratio coefficient
    const DomainMappedScalarCoefficient& poissonRatio() const { return poissonRatio_; }
    
    // =========================================================================
    // Boundary condition interfaces (use Coefficient)
    // =========================================================================
    
    /// Add fixed displacement boundary condition (batch setting)
    void addFixedDisplacementBC(const std::set<int>& boundaryIds, 
                                 const VectorCoefficient* displacement);
    
    /// Clear boundary conditions
    void clearBoundaryConditions() { 
        displacementBCs_.clear(); 
    }
    
    // =========================================================================
    // Thermal expansion coefficient interface
    // =========================================================================
    
    /// Set thermal expansion coefficient for specified domains
    void setThermalExpansion(const std::set<int>& domains, const Coefficient* alphaT);
    
    /// Get thermal expansion coefficient
    const DomainMappedScalarCoefficient& thermalExpansion() const { return thermalExpansion_; }
    
    /// Check if thermal expansion load exists
    bool hasThermalExpansion() const { return !thermalExpansion_.empty(); }
    
    // =========================================================================
    // Solve interface
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return fieldValues_->current(FieldId::Displacement); }
    GridFunction& field() override { return fieldValues_->current(FieldId::Displacement); }

private:
    FieldValues* fieldValues_ = nullptr;  ///< Non-owning reference to field manager
    
    DomainMappedScalarCoefficient youngModulus_;
    DomainMappedScalarCoefficient poissonRatio_;
    DomainMappedScalarCoefficient thermalExpansion_;  ///< Thermal expansion coefficient (domain mapped)
    
    std::map<int, const VectorCoefficient*> displacementBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP

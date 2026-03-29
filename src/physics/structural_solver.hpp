#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <set>

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
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Structural"; }
    FieldId fieldId() const override { return FieldId::Displacement; }
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialDisplacement = 0.0);
    
    // Material coefficients
    void setYoungModulus(const std::set<int>& domains, const Coefficient* E);
    void setYoungModulus(const Coefficient* E) { youngModulus_.setAll(E); }
    
    void setPoissonRatio(const std::set<int>& domains, const Coefficient* nu);
    void setPoissonRatio(const Coefficient* nu) { poissonRatio_.setAll(nu); }
    
    const DomainMappedScalarCoefficient& youngModulus() const { return youngModulus_; }
    const DomainMappedScalarCoefficient& poissonRatio() const { return poissonRatio_; }
    
    // Boundary conditions
    void addFixedDisplacementBC(const std::set<int>& boundaryIds, const VectorCoefficient* displacement);
    void clearBoundaryConditions() { displacementBCs_.clear(); }
    
    // Thermal expansion
    void setThermalExpansion(const std::set<int>& domains, const MatrixCoefficient* alphaT);
    const DomainMappedMatrixCoefficient& thermalExpansion() const { return thermalExpansion_; }
    bool hasThermalExpansion() const { return !thermalExpansion_.empty(); }
    
    void assemble() override;

private:
    DomainMappedScalarCoefficient youngModulus_;
    DomainMappedScalarCoefficient poissonRatio_;
    DomainMappedMatrixCoefficient thermalExpansion_;
    std::map<int, const VectorCoefficient*> displacementBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP
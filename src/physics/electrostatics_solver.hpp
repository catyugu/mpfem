#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <set>

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
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "Electrostatics"; }
    FieldId fieldId() const override { return FieldId::ElectricPotential; }
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialPotential = 0.0);
    
    // Material coefficients - matrix form for anisotropic conductivity
    void setConductivity(const std::set<int>& domains, const MatrixCoefficient* sigma);
    void setConductivity(const MatrixCoefficient* sigma) { conductivity_.setAll(sigma); }
    const DomainMappedMatrixCoefficient& conductivity() const { return conductivity_; }
    
    // Boundary conditions
    void addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage);
    void clearBoundaryConditions() { voltageBCs_.clear(); }
    
    void assemble() override;

private:
    DomainMappedMatrixCoefficient conductivity_;
    std::map<int, const Coefficient*> voltageBCs_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
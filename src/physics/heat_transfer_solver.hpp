#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <set>

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
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "HeatTransfer"; }
    FieldId fieldId() const override { return FieldId::Temperature; }
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues);
    
    // Material coefficients - matrix form for anisotropic conductivity
    void setConductivity(const std::set<int>& domains, const MatrixCoefficient* k);
    void setConductivity(const MatrixCoefficient* k) { conductivity_.setAll(k); }
    const DomainMappedMatrixCoefficient& conductivity() const { return conductivity_; }
    
    void setHeatSource(const std::set<int>& domains, const Coefficient* Q);
    void setHeatSource(const Coefficient* Q) { heatSource_.setAll(Q); }
    const DomainMappedScalarCoefficient& heatSource() const { return heatSource_; }
    
    // Boundary conditions
    void addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature);
    void addConvectionBC(const std::set<int>& boundaryIds, const Coefficient* h, const Coefficient* Tinf);
    void clearBoundaryConditions() { temperatureBCs_.clear(); convBCs_.clear(); }
    
    void assemble() override;

private:
    struct ConvBC { const Coefficient* h; const Coefficient* Tinf; };
    
    DomainMappedMatrixCoefficient conductivity_;
    DomainMappedScalarCoefficient heatSource_;
    std::map<int, const Coefficient*> temperatureBCs_;
    std::map<int, ConvBC> convBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP
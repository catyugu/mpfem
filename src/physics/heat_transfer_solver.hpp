#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <vector>
#include <memory>
#include <set>

namespace mpfem {

/**
 * @brief Heat transfer solver
 * 
 * Solves: -div(k * grad T) = Q
 * 
 * Design principles:
 * - Single-field solver does not contain coupling logic
 * - Coefficient supports domain selection, each domain can use different coefficients
 * - Boundary conditions use Coefficient instead of direct values
 * - Field values are owned by FieldValues, solver holds reference
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "Temperature"; }
    
    /// Initialize solver (creates FESpace, registers field with FieldValues)
    /// @param mesh The mesh
    /// @param fieldValues Field value manager (solver registers its field here)
    bool initialize(const Mesh& mesh, FieldValues& fieldValues);
    
    // =========================================================================
    // Material coefficient interfaces (support domain selection)
    // =========================================================================
    
    /// Set thermal conductivity coefficient for specified domains
    void setConductivity(const std::set<int>& domains, const Coefficient* k);
    
    /// Set thermal conductivity coefficient for all domains (for coupling)
    void setConductivity(const Coefficient* k) {
        conductivity_.setAll(k);
    }
    
    /// Get thermal conductivity coefficient
    const DomainMappedScalarCoefficient& conductivity() const { return conductivity_; }
    
    /// Set heat source coefficient for specified domains
    void setHeatSource(const std::set<int>& domains, const Coefficient* Q);
    
    /// Set heat source coefficient for all domains (for coupling coefficient)
    void setHeatSource(const Coefficient* Q) {
        heatSource_.setAll(Q);
    }
    
    /// Get heat source coefficient
    const DomainMappedScalarCoefficient& heatSource() const { return heatSource_; }
    
    // =========================================================================
    // Boundary condition interfaces (use Coefficient)
    // =========================================================================
    
    /// Add temperature boundary condition (batch setting)
    void addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature);
    
    /// Add convection boundary condition: h*(T - Tinf)
    void addConvectionBC(const std::set<int>& boundaryIds, 
                         const Coefficient* h, 
                         const Coefficient* Tinf);
    
    /// Clear boundary conditions
    void clearBoundaryConditions() { 
        temperatureBCs_.clear(); 
        convBCs_.clear(); 
    }
    
    // =========================================================================
    // Solve interface
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return fieldValues_->current(FieldId::Temperature); }
    GridFunction& field() override { return fieldValues_->current(FieldId::Temperature); }

private:
    struct ConvBC { 
        const Coefficient* h;
        const Coefficient* Tinf;
    };
    
    FieldValues* fieldValues_ = nullptr;  ///< Non-owning reference to field manager
    DomainMappedScalarCoefficient conductivity_;
    DomainMappedScalarCoefficient heatSource_;
    
    std::map<int, const Coefficient*> temperatureBCs_;
    std::map<int, ConvBC> convBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP
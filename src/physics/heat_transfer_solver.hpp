#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/assembler.hpp"
#include "assembly/integrators.hpp"
#include "solver/solver_factory.hpp"
#include "mesh/mesh_topology.hpp"
#include <memory>
#include <vector>
#include <map>

namespace mpfem {

/**
 * @file heat_transfer_solver.hpp
 * @brief Solver for steady-state heat transfer problems.
 * 
 * Solves the steady-state heat equation:
 *   -∇·(k∇T) = Q  in Ω
 * with boundary conditions:
 *   T = T₀ on Γ_D (temperature boundary)
 *   n·(k∇T) = h(T - T∞) on Γ_C (convection boundary)
 *   n·(k∇T) = 0 on Γ_N (insulation boundary)
 * 
 * where:
 * - T is the temperature field
 * - k is the thermal conductivity
 * - Q is the volumetric heat source (e.g., Joule heating)
 * - h is the convection coefficient
 * - T∞ is the ambient temperature
 */

/**
 * @brief Heat transfer solver for steady-state thermal analysis.
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    
    HeatTransferSolver() = default;
    
    explicit HeatTransferSolver(int order) {
        order_ = order;
    }
    
    ~HeatTransferSolver() override = default;
    
    // -------------------------------------------------------------------------
    // PhysicsFieldSolver interface
    // -------------------------------------------------------------------------
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    
    std::string fieldName() const override { return "Temperature"; }
    
    bool initialize(const Mesh& mesh, 
                   const PWConstCoefficient& thermalConductivity) override;
    
    void addDirichletBC(int boundaryId, Real value) override {
        dirichletBCs_[boundaryId] = std::make_shared<ConstantCoefficient>(value);
    }
    
    void addDirichletBC(int boundaryId, std::shared_ptr<Coefficient> coef) override {
        dirichletBCs_[boundaryId] = std::move(coef);
    }
    
    void clearBoundaryConditions() override {
        dirichletBCs_.clear();
        convectionBCs_.clear();
    }
    
    // -------------------------------------------------------------------------
    // Convection boundary conditions (Robin BC)
    // -------------------------------------------------------------------------
    
    /**
     * @brief Add a convection boundary condition.
     * @param boundaryId Boundary attribute ID
     * @param h Convection coefficient
     * @param Tinf Ambient temperature
     */
    void addConvectionBC(int boundaryId, Real h, Real Tinf) {
        ConvectionBC bc;
        bc.h = std::make_shared<ConstantCoefficient>(h);
        bc.Tinf = std::make_shared<ConstantCoefficient>(Tinf);
        convectionBCs_[boundaryId] = bc;
    }
    
    /**
     * @brief Add a convection boundary condition with coefficient.
     * @param boundaryId Boundary attribute ID
     * @param h Convection coefficient (as Coefficient)
     * @param Tinf Ambient temperature (as Coefficient)
     */
    void addConvectionBC(int boundaryId, std::shared_ptr<Coefficient> h,
                         std::shared_ptr<Coefficient> Tinf) {
        ConvectionBC bc;
        bc.h = h;
        bc.Tinf = Tinf;
        convectionBCs_[boundaryId] = bc;
    }
    
    // -------------------------------------------------------------------------
    // Heat source
    // -------------------------------------------------------------------------
    
    /**
     * @brief Set volumetric heat source coefficient.
     * @param Q Heat source coefficient (W/m³)
     */
    void setHeatSource(std::shared_ptr<Coefficient> Q) {
        heatSource_ = Q;
    }
    
    /**
     * @brief Set Joule heating from electric field.
     * @param V Electric potential field
     * @param sigma Electrical conductivity (base class pointer)
     */
    void setJouleHeating(const GridFunction* V, const Coefficient* sigma);
    
    // -------------------------------------------------------------------------
    // Assembly and solve
    // -------------------------------------------------------------------------
    
    void assemble() override;
    
    bool solve() override;
    
    // -------------------------------------------------------------------------
    // Results access
    // -------------------------------------------------------------------------
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }
    
    const FESpace& feSpace() const override { return *fes_; }
    
    Index numDofs() const override { return fes_ ? fes_->numDofs() : 0; }
    
    Real minValue() const override {
        return T_ ? T_->values().minCoeff() : 0.0;
    }
    
    Real maxValue() const override {
        return T_ ? T_->values().maxCoeff() : 0.0;
    }
    
    // -------------------------------------------------------------------------
    // Temperature-dependent thermal conductivity
    // -------------------------------------------------------------------------
    
    /**
     * @brief Set temperature field for temperature-dependent thermal conductivity.
     * @param temperature Temperature field (GridFunction)
     */
    void setTemperatureField(const GridFunction* temperature) {
        temperatureField_ = temperature;
    }
    
    /**
     * @brief Update thermal conductivity based on temperature field.
     */
    void updateThermalConductivity();
    
    // -------------------------------------------------------------------------
    // Access to internal coefficients
    // -------------------------------------------------------------------------
    
    std::shared_ptr<PWConstCoefficient> thermalConductivity() const { 
        return thermalConductivity_; 
    }
    
private:
    struct ConvectionBC {
        std::shared_ptr<Coefficient> h;
        std::shared_ptr<Coefficient> Tinf;
    };
    
    void applyDirichletBCs();
    void assembleConvectionBCs();
    
    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------
    
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> T_;
    std::shared_ptr<PWConstCoefficient> thermalConductivity_;
    
    std::unique_ptr<BilinearFormAssembler> bilinearAsm_;
    std::unique_ptr<LinearFormAssembler> linearAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    std::map<int, std::shared_ptr<Coefficient>> dirichletBCs_;
    std::map<int, ConvectionBC> convectionBCs_;
    std::shared_ptr<Coefficient> heatSource_;
    
    const GridFunction* temperatureField_ = nullptr;
};

// =============================================================================
// Joule Heat Source Coefficient (Improved)
// =============================================================================

/**
 * @brief Coefficient that computes Joule heating Q = σ|∇V|².
 * 
 * Used for electro-thermal coupling. Evaluates the electric field
 * gradient and computes the volumetric heat source.
 */
class JouleHeatCoefficient : public Coefficient {
public:
    JouleHeatCoefficient() = default;
    
    /**
     * @brief Construct from potential field and conductivity.
     * @param V Electric potential field
     * @param sigma Electrical conductivity
     */
    JouleHeatCoefficient(const GridFunction* V, 
                         const Coefficient* sigma)
        : V_(V), sigma_(sigma) {}
    
    /**
     * @brief Set electric potential field.
     */
    void setPotential(const GridFunction* V) { V_ = V; }
    
    /**
     * @brief Set electrical conductivity.
     */
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    const GridFunction* V_ = nullptr;
    const Coefficient* sigma_ = nullptr;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP

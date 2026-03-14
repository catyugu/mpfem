#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Structural mechanics solver for linear elasticity.
 * 
 * Solves: -div(sigma) = 0
 * where sigma = C : (epsilon - epsilon_thermal)
 * and epsilon_thermal = alpha_T * (T - T_ref) * I
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// Initialize with isotropic material properties
    /// @param E Young's modulus [Pa]
    /// @param nu Poisson's ratio [-]
    bool initialize(const Mesh& mesh,
                    const PWConstCoefficient& youngModulus,
                    const PWConstCoefficient& poissonRatio);
    
    void addDirichletBC(int boundaryId, Real value) override {
        // For displacement, we store vector BCs separately
        // This is a simplified interface for scalar BC
        bcValues_[boundaryId] = Vector3(value, value, value);
    }
    
    /// Add Dirichlet BC with vector value
    void addDirichletBC(int boundaryId, const Vector3& disp) {
        bcValues_[boundaryId] = disp;
    }
    
    /// Add Dirichlet BC for specific component (0=x, 1=y, 2=z)
    void addDirichletBC(int boundaryId, int component, Real value) {
        componentBCs_[boundaryId * 3 + component] = value;
    }
    
    void clearBoundaryConditions() override { 
        bcValues_.clear(); 
        componentBCs_.clear();
    }
    
    /// Set thermal strain coefficient: alpha_T * (T - T_ref)
    /// This is a vector coefficient (3 components for 3D)
    void setThermalStrain(const VectorCoefficient* thermalStrain) {
        thermalStrain_ = thermalStrain;
    }
    
    /// Set temperature field for thermal expansion (non-owning)
    void setTemperatureField(const GridFunction* T) { T_ = T; }
    
    /// Set reference temperature for thermal expansion
    void setReferenceTemperature(Real Tref) { Tref_ = Tref; }
    
    /// Set thermal expansion coefficient (non-owning)
    void setThermalExpansion(const Coefficient* alphaT) { alphaT_ = alphaT; }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *u_; }
    GridFunction& field() override { return *u_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    
    /// Get stress field (computed after solve)
    const GridFunction& stress() const { return *stress_; }
    
    /// Get strain field (computed after solve)
    const GridFunction& strain() const { return *strain_; }
    
private:
    void computeStressStrain();
    
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> u_;        // Displacement
    std::unique_ptr<GridFunction> stress_;   // Stress (6 components)
    std::unique_ptr<GridFunction> strain_;   // Strain (6 components)
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient EInternal_;
    PWConstCoefficient nuInternal_;
    
    const GridFunction* T_ = nullptr;        // Temperature field
    const Coefficient* alphaT_ = nullptr;    // Thermal expansion coefficient
    const VectorCoefficient* thermalStrain_ = nullptr;
    Real Tref_ = 293.15;                     // Reference temperature
    
    std::map<int, Vector3> bcValues_;
    std::map<int, Real> componentBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP

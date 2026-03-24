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
    
    bool initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialTemperature = 293.15);
    
    // Material coefficients - matrix form for anisotropic thermal conductivity
    void setThermalConductivity(const std::set<int>& domains, const MatrixCoefficient* k);
    void setThermalConductivity(const MatrixCoefficient* k) { conductivity_.setAll(k); }
    const DomainMappedMatrixCoefficient& thermalConductivity() const { return conductivity_; }
    
    void setHeatSource(const std::set<int>& domains, const Coefficient* Q);
    void setHeatSource(const Coefficient* Q) { heatSource_.setAll(Q); }
    const DomainMappedScalarCoefficient& heatSource() const { return heatSource_; }
    
    // Material properties for transient analysis (rho * Cp)
    void setDensity(const std::set<int>& domains, const Coefficient* rho);
    void setDensity(const Coefficient* rho) { density_.setAll(rho); }
    const DomainMappedScalarCoefficient& density() const { return density_; }
    
    void setSpecificHeat(const std::set<int>& domains, const Coefficient* Cp);
    void setSpecificHeat(const Coefficient* Cp) { specificHeat_.setAll(Cp); }
    const DomainMappedScalarCoefficient& specificHeat() const { return specificHeat_; }
    
    // Mass matrix for transient terms: M = ∫ ρCp φᵢ φⱼ dΩ
    void assembleMassMatrix();
    const SparseMatrix& massMatrix() const { return massMatrix_; }
    bool massMatrixAssembled() const { return massMatrixAssembled_; }
    
    // Boundary conditions
    void addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature);
    void addConvectionBC(const std::set<int>& boundaryIds, const Coefficient* h, const Coefficient* Tinf);
    void clearBoundaryConditions() { temperatureBCs_.clear(); convBCs_.clear(); }
    
    void assemble() override;
    
    /**
     * @brief Solve a custom linear system Ax = b with applied boundary conditions
     * 
     * This is useful for time integrators like BDF1 that need to solve
     * a modified system (e.g., M + dt*K) rather than the standard K.
     * 
     * @param A System matrix
     * @param x Solution vector (output)
     * @param b Right-hand side vector
     * @return true if solved successfully
     */
    bool solveLinearSystem(const SparseMatrix& A, Vector& x, const Vector& b);
    
    /**
     * @brief Get stiffness matrix before BC application
     * 
     * This is needed for transient time integrators like BDF1 that need
     * to form the combined system (M + dt*K) before BCs are applied.
     * 
     * @return Stiffness matrix K (without BC modifications)
     */
    const SparseMatrix& stiffnessMatrixBeforeBC() const { return stiffnessMatrixBeforeBC_; }
    
    /**
     * @brief Get RHS vector before BC application
     * 
     * This is needed for transient time integrators like BDF1 that need
     * to form the combined RHS (M*T_prev + dt*Q) before BCs are applied.
     * 
     * @return RHS vector Q (without BC modifications)
     */
    const Vector& rhsBeforeBC() const { return rhsBeforeBC_; }

private:
    struct ConvBC { const Coefficient* h; const Coefficient* Tinf; };
    
    DomainMappedMatrixCoefficient conductivity_;
    DomainMappedScalarCoefficient heatSource_;
    DomainMappedScalarCoefficient density_;
    DomainMappedScalarCoefficient specificHeat_;
    std::unique_ptr<ProductCoefficient> rhoCp_;  ///< Product coefficient for mass matrix (rho*Cp), stored as member to avoid dangling pointer
    SparseMatrix massMatrix_;
    bool massMatrixAssembled_ = false;
    std::map<int, const Coefficient*> temperatureBCs_;
    std::map<int, ConvBC> convBCs_;
    
    /// @brief Stiffness matrix before BC application (for transient time integrators)
    SparseMatrix stiffnessMatrixBeforeBC_;
    
    /// @brief RHS vector before BC application (for transient time integrators)
    Vector rhsBeforeBC_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP
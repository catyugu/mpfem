#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Electrostatics solver - minimal design for single-field analysis.
 * 
 * Design principle: Single-field solver should NOT contain coupling logic.
 * Temperature-dependent conductivity should be handled externally via setConductivity().
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    bool initialize(const Mesh& mesh, 
                    const PWConstCoefficient& conductivity) override;
    
    void addDirichletBC(int boundaryId, Real value) override {
        bcValues_[boundaryId] = value;
    }
    
    void clearBoundaryConditions() override { bcValues_.clear(); }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *V_; }
    GridFunction& field() override { return *V_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    
    /// Set conductivity coefficient (non-owning pointer)
    /// This allows external coupling modules to provide temperature-dependent conductivity
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// Get conductivity coefficient
    const Coefficient* conductivity() const { return sigma_; }
    
private:
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> V_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient sigmaInternal_;
    const Coefficient* sigma_ = nullptr;
    
    std::map<int, Real> bcValues_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP

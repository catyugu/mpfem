#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Heat transfer solver - minimal design for single-field analysis.
 * 
 * Design principle: Single-field solver should NOT contain coupling logic.
 * Coupling (like Joule heating) should be handled externally.
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "Temperature"; }
    
    bool initialize(const Mesh& mesh, 
                    const PWConstCoefficient& conductivity) override;
    
    void addDirichletBC(int bid, Real val) override { bcValues_[bid] = val; }
    void clearBoundaryConditions() override { bcValues_.clear(); convBCs_.clear(); }
    
    /// Add convection boundary condition
    void addConvectionBC(int bid, Real h, Real Tinf) {
        convBCs_[bid] = {h, Tinf};
    }
    
    /// Set heat source coefficient (non-owning pointer)
    void setHeatSource(const Coefficient* Q) { heatSource_ = Q; }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    
private:
    struct ConvBC { Real h, Tinf; };
    
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> T_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient kInternal_;
    const Coefficient* k_ = nullptr;
    const Coefficient* heatSource_ = nullptr;
    
    std::map<int, Real> bcValues_;
    std::map<int, ConvBC> convBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP
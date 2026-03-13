#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief 静电场求解器 - 最小化设计
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
    
    /// 设置电导率（非拥有指针）
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// 获取电导率
    const Coefficient* conductivity() const { return sigma_; }
    
    /// 设置温度依赖电导率参数
    void setTempDepSigma(int domainId, Real rho0, Real alpha, Real tref, Real sigma0) {
        if (!tempDepSigma_) {
            tempDepSigma_ = std::make_unique<TemperatureDependentConductivity>();
        }
        tempDepSigma_->setMaterial(domainId, rho0, alpha, tref, sigma0);
        sigma_ = tempDepSigma_.get();
    }
    
    /// 设置温度场（用于温度依赖电导率）
    void setTemperatureField(const GridFunction* T) {
        if (tempDepSigma_) {
            tempDepSigma_->setTemperatureField(T);
        }
    }
    
    /// 是否启用了温度依赖电导率
    bool hasTempDepSigma() const { return tempDepSigma_ != nullptr; }
    
private:
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> V_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient sigmaInternal_;
    std::unique_ptr<TemperatureDependentConductivity> tempDepSigma_;
    const Coefficient* sigma_ = nullptr;
    
    std::map<int, Real> bcValues_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
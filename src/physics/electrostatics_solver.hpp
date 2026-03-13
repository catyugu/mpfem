#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/// 温度依赖电导率系数
class TempDepSigmaCoefficient : public Coefficient {
public:
    void setMaterial(int domainId, Real rho0, Real alpha, Real tref, Real sigma0) {
        ensureSize(domainId);
        rho0_[domainId - 1] = rho0;
        alpha_[domainId - 1] = alpha;
        tref_[domainId - 1] = tref;
        sigma0_[domainId - 1] = sigma0;
    }
    
    void setTemperatureField(const GridFunction* T) { T_ = T; }
    
    Real eval(ElementTransform& trans) const override;
    
private:
    void ensureSize(int domainId) {
        if (static_cast<int>(rho0_.size()) < domainId) {
            rho0_.resize(domainId, 0.0);
            alpha_.resize(domainId, 0.0);
            tref_.resize(domainId, 293.15);
            sigma0_.resize(domainId, 0.0);
        }
    }
    
    std::vector<Real> rho0_;
    std::vector<Real> alpha_;
    std::vector<Real> tref_;
    std::vector<Real> sigma0_;
    const GridFunction* T_ = nullptr;
};

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
            tempDepSigma_ = std::make_unique<TempDepSigmaCoefficient>();
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
    std::unique_ptr<TempDepSigmaCoefficient> tempDepSigma_;
    const Coefficient* sigma_ = nullptr;
    
    std::map<int, Real> bcValues_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
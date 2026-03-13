#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief 热传导求解器 - 最小化设计
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
    void addDirichletBC(int, std::shared_ptr<Coefficient>) override {}
    void clearBoundaryConditions() override { bcValues_.clear(); convBCs_.clear(); }
    
    /// 添加对流边界条件
    void addConvectionBC(int bid, Real h, Real Tinf) {
        convBCs_[bid] = {h, Tinf};
    }
    
    /// 设置热源（非拥有指针）
    void setHeatSource(const Coefficient* Q) { heatSource_ = Q; }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    Real minValue() const override { return T_->values().minCoeff(); }
    Real maxValue() const override { return T_->values().maxCoeff(); }
    
private:
    void applyBCs();
    
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

/// 焦耳热系数（解耦设计）
class JouleHeatCoefficient : public Coefficient {
public:
    using GradientFunc = std::function<Vector3(int, const Real*, ElementTransform&)>;
    using ConductivityFunc = std::function<Real(ElementTransform&)>;
    
    void setGradientFunc(GradientFunc f) { gradFunc_ = std::move(f); }
    void setConductivityFunc(ConductivityFunc f) { sigmaFunc_ = std::move(f); }
    
    Real eval(ElementTransform& trans) const override {
        if (!gradFunc_ || !sigmaFunc_) return 0.0;
        auto g = gradFunc_(trans.elementIndex(), &trans.integrationPoint().xi, trans);
        return sigmaFunc_(trans) * g.squaredNorm();
    }
    
private:
    GradientFunc gradFunc_;
    ConductivityFunc sigmaFunc_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP

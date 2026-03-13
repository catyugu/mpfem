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

/// 焦耳热系数（优化设计：直接持有指针，避免 std::function 开销）
class JouleHeatCoefficient : public Coefficient {
public:
    /// 设置电势场（非拥有指针）
    void setPotential(const GridFunction* V) { V_ = V; }
    
    /// 设置电导率系数（非拥有指针）
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    Real eval(ElementTransform& trans) const override {
        if (!V_ || !sigma_) return 0.0;
        // 关键：先计算sigma，再计算梯度
        // 因为gradient()会调用setIntegrationPoint改变trans状态
        Real sigma_val = sigma_->eval(trans);
        Vector3 g = V_->gradient(trans.elementIndex(), &trans.integrationPoint().xi, trans);
        return sigma_val * g.squaredNorm();
    }
    
private:
    const GridFunction* V_ = nullptr;
    const Coefficient* sigma_ = nullptr;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP

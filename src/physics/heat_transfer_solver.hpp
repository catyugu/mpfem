#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include <map>

namespace mpfem {

/**
 * @brief 热传导求解器
 * 
 * 求解：-div(k * grad T) = Q
 * 
 * 设计原则：单场求解器不包含耦合逻辑。
 * 焦耳热等耦合热源应通过setHeatSource()由外部设置。
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "Temperature"; }
    
    /// 初始化求解器
    /// @param mesh 网格
    /// @param conductivity 热导率系数
    bool initialize(const Mesh& mesh, const Coefficient& conductivity);
    
    void addDirichletBC(int bid, Real val) { bcValues_[bid] = val; }
    void clearBoundaryConditions() { bcValues_.clear(); convBCs_.clear(); }
    
    /// 添加对流边界条件: h*(T - Tinf)
    void addConvectionBC(int bid, Real h, Real Tinf) {
        convBCs_[bid] = {h, Tinf};
    }
    
    /// 设置热源系数（非拥有指针）
    void setHeatSource(const Coefficient* Q) { heatSource_ = Q; }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }
    
private:
    struct ConvBC { Real h, Tinf; };
    
    std::unique_ptr<GridFunction> T_;
    const Coefficient* k_ = nullptr;
    const Coefficient* heatSource_ = nullptr;
    
    std::map<int, Real> bcValues_;
    std::map<int, ConvBC> convBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP
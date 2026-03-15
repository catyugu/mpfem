#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include <map>
#include <vector>

namespace mpfem {

/**
 * @brief 静电场求解器
 * 
 * 求解：-div(sigma * grad V) = 0
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 电导率系数由外部设置，求解器持有非拥有引用
 * - 边界条件接口具有物理意义
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    /// 初始化求解器（不设置材料系数，需单独调用 setConductivity）
    /// @param mesh 网格
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 边界条件接口（物理语义）
    // =========================================================================
    
    /// 添加电压边界条件
    /// @param boundaryId 边界ID
    /// @param voltage 电压值 (V)
    void addVoltageBC(int boundaryId, Real voltage) {
        voltageBCs_[boundaryId] = voltage;
    }
    
    /// 批量设置电压边界条件
    void setVoltageBCs(const std::vector<std::pair<int, Real>>& bcs) {
        for (const auto& [id, val] : bcs) {
            voltageBCs_[id] = val;
        }
    }
    
    /// 清除边界条件
    void clearBoundaryConditions() { voltageBCs_.clear(); }
    
    // =========================================================================
    // 材料系数接口
    // =========================================================================
    
    /// 设置电导率系数（非拥有指针，生命周期由调用者管理）
    /// @param sigma 电导率系数
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// 获取电导率系数
    const Coefficient* conductivity() const { return sigma_; }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *V_; }
    GridFunction& field() override { return *V_; }
    
private:
    std::unique_ptr<GridFunction> V_;
    const Coefficient* sigma_ = nullptr;      ///< 电导率系数（非拥有）
    std::map<int, Real> voltageBCs_;          ///< 边界ID -> 电压值
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP

#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <vector>
#include <memory>

namespace mpfem {

/**
 * @brief 热传导求解器
 * 
 * 求解：-div(k * grad T) = Q
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 热导率系数由外部设置，求解器持有非拥有引用
 * - 边界条件接口具有物理意义
 * - 边界条件系数由求解器内部持有
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "Temperature"; }
    
    /// 初始化求解器（不设置材料系数，需单独调用 setConductivity）
    /// @param mesh 网格
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 边界条件接口（物理语义）
    // =========================================================================
    
    /// 添加温度边界条件
    /// @param boundaryId 边界ID
    /// @param temperature 温度值 (K)
    void addTemperatureBC(int boundaryId, Real temperature) {
        temperatureBCs_[boundaryId] = temperature;
    }
    
    /// 批量设置温度边界条件
    void setTemperatureBCs(const std::vector<std::pair<int, Real>>& bcs) {
        for (const auto& [id, val] : bcs) {
            temperatureBCs_[id] = val;
        }
    }
    
    /// 添加对流边界条件: h*(T - Tinf)
    /// @param boundaryId 边界ID
    /// @param h 对流换热系数 (W/(m²·K))
    /// @param Tinf 环境温度 (K)
    void addConvectionBC(int boundaryId, Real h, Real Tinf) {
        convBCs_[boundaryId] = {h, Tinf};
    }
    
    /// 清除边界条件
    void clearBoundaryConditions() { 
        temperatureBCs_.clear(); 
        convBCs_.clear(); 
    }
    
    // =========================================================================
    // 材料系数接口
    // =========================================================================
    
    /// 设置热导率系数（非拥有指针，生命周期由调用者管理）
    void setConductivity(const Coefficient* k) { k_ = k; }
    
    /// 获取热导率系数
    const Coefficient* conductivity() const { return k_; }
    
    /// 设置热源系数（非拥有指针，生命周期由调用者管理）
    void setHeatSource(const Coefficient* Q) { heatSource_ = Q; }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }
    
private:
    struct ConvBC { 
        Real h;      ///< 对流换热系数
        Real Tinf;   ///< 环境温度
    };
    
    std::unique_ptr<GridFunction> T_;
    const Coefficient* k_ = nullptr;           ///< 热导率系数（非拥有）
    const Coefficient* heatSource_ = nullptr;  ///< 热源系数（非拥有）
    
    std::map<int, Real> temperatureBCs_;       ///< 边界ID -> 温度值
    std::map<int, ConvBC> convBCs_;            ///< 对流边界条件
    
    // 内部持有的边界条件系数（拥有所有权）
    std::vector<std::unique_ptr<ConstantCoefficient>> ownedConvH_;
    std::vector<std::unique_ptr<ConstantCoefficient>> ownedConvTinf_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP

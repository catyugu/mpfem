#ifndef MPFEM_HEAT_TRANSFER_SOLVER_HPP
#define MPFEM_HEAT_TRANSFER_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <vector>
#include <memory>
#include <set>

namespace mpfem {

/**
 * @brief 热传导求解器
 * 
 * 求解：-div(k * grad T) = Q
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 系数支持域选择，每个域可以使用不同的系数
 * - 边界条件使用 Coefficient 而非直接数值
 */
class HeatTransferSolver : public PhysicsFieldSolver {
public:
    HeatTransferSolver() = default;
    explicit HeatTransferSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Temperature; }
    std::string fieldName() const override { return "Temperature"; }
    
    /// 初始化求解器
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 材料系数接口（支持域选择）
    // =========================================================================
    
    /// 设置指定域的热导率系数
    void setConductivity(const std::set<int>& domains, const Coefficient* k);
    
    /// 设置所有域的热导率系数
    void setConductivity(const Coefficient* k) {
        conductivity_.setAll(k);
    }
    
    /// 获取热导率系数
    const DomainMappedCoefficient& conductivity() const { return conductivity_; }
    
    /// 设置指定域的热源系数
    void setHeatSource(const std::set<int>& domains, const Coefficient* Q);
    
    /// 设置所有域的热源系数
    void setHeatSource(const Coefficient* Q) {
        heatSource_.setAll(Q);
    }
    
    /// 获取热源系数
    const DomainMappedCoefficient& heatSource() const { return heatSource_; }
    
    // =========================================================================
    // 边界条件接口（使用 Coefficient）
    // =========================================================================
    
    /// 添加温度边界条件（批量设置）
    void addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature);
    
    /// 添加对流边界条件: h*(T - Tinf)
    void addConvectionBC(const std::set<int>& boundaryIds, 
                         const Coefficient* h, 
                         const Coefficient* Tinf);
    
    /// 清除边界条件
    void clearBoundaryConditions() { 
        temperatureBCs_.clear(); 
        convBCs_.clear(); 
    }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *T_; }
    GridFunction& field() override { return *T_; }

private:
    struct ConvBC { 
        const Coefficient* h;
        const Coefficient* Tinf;
    };
    
    std::unique_ptr<GridFunction> T_;
    DomainMappedCoefficient conductivity_;
    DomainMappedCoefficient heatSource_;
    
    std::map<int, const Coefficient*> temperatureBCs_;
    std::map<int, ConvBC> convBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_HEAT_TRANSFER_SOLVER_HPP

#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include <map>
#include <vector>
#include <set>

namespace mpfem {

/**
 * @brief 静电场求解器
 * 
 * 求解：-div(sigma * grad V) = 0
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 电导率系数支持域选择，每个域可以使用不同的系数
 * - 边界条件使用 Coefficient 而非直接数值
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    /// 初始化求解器
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 材料系数接口（支持域选择）
    // =========================================================================
    
    /// 设置指定域的电导率系数（非拥有指针，覆盖已存在的）
    void setConductivity(const std::set<int>& domains, const Coefficient* sigma);
    
    /// 设置所有域的电导率系数（非拥有指针，用于耦合系数）
    void setConductivity(const Coefficient* sigma) {
        conductivity_.setAll(sigma);
    }
    
    /// 获取电导率系数
    const DomainMappedCoefficient& conductivity() const { return conductivity_; }
    
    // =========================================================================
    // 边界条件接口（使用 Coefficient）
    // =========================================================================
    
    /// 添加电压边界条件（批量设置，使用 Coefficient）
    void addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage);
    
    /// 清除边界条件
    void clearBoundaryConditions() { voltageBCs_.clear(); }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *V_; }
    GridFunction& field() override { return *V_; }

private:
    std::unique_ptr<GridFunction> V_;
    DomainMappedCoefficient conductivity_;
    std::map<int, const Coefficient*> voltageBCs_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP

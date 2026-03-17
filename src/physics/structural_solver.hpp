#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/integrator.hpp"
#include <vector>
#include <set>

namespace mpfem {

/**
 * @brief 结构力学求解器（线弹性）
 * 
 * 求解: -div(sigma) = 0
 * 其中 sigma = C : epsilon
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 材料系数支持域选择
 * - 边界条件使用 Coefficient
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// 初始化求解器
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 材料系数接口（支持域选择）
    // =========================================================================
    
    /// 设置指定域的杨氏模量系数
    void setYoungModulus(const std::set<int>& domains, const Coefficient* E);
    
    /// 设置所有域的杨氏模量系数
    void setYoungModulus(const Coefficient* E) {
        youngModulus_.setAll(E);
    }
    
    /// 设置指定域的泊松比系数
    void setPoissonRatio(const std::set<int>& domains, const Coefficient* nu);
    
    /// 设置所有域的泊松比系数
    void setPoissonRatio(const Coefficient* nu) {
        poissonRatio_.setAll(nu);
    }
    
    /// 获取杨氏模量系数
    const DomainMappedCoefficient& youngModulus() const { return youngModulus_; }
    
    /// 获取泊松比系数
    const DomainMappedCoefficient& poissonRatio() const { return poissonRatio_; }
    
    // =========================================================================
    // 边界条件接口（使用 Coefficient）
    // =========================================================================
    
    /// 添加固定位移边界条件（批量设置）
    void addFixedDisplacementBC(const std::set<int>& boundaryIds, 
                                 const VectorCoefficient* displacement);
    
    /// 清除边界条件
    void clearBoundaryConditions() { 
        displacementBCs_.clear(); 
    }
    
    // =========================================================================
    // 耦合载荷接口
    // =========================================================================
    
    /// 添加线性积分器（用于耦合载荷，如热膨胀）
    void addLinearIntegrator(std::unique_ptr<VectorDomainLinearIntegrator> integrator) {
        linearIntegrators_.push_back(std::move(integrator));
    }
    
    /// 清除线性积分器
    void clearLinearIntegrators() { linearIntegrators_.clear(); }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *u_; }
    GridFunction& field() override { return *u_; }

private:
    std::unique_ptr<GridFunction> u_;        ///< 位移场
    
    DomainMappedCoefficient youngModulus_;
    DomainMappedCoefficient poissonRatio_;
    
    std::vector<std::unique_ptr<VectorDomainLinearIntegrator>> linearIntegrators_;
    
    std::map<int, const VectorCoefficient*> displacementBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP
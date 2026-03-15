#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/integrator.hpp"
#include <vector>

namespace mpfem {

/**
 * @brief 结构力学求解器（线弹性）
 * 
 * 求解: -div(sigma) = 0
 * 其中 sigma = C : epsilon
 * 
 * 设计原则：
 * - 单场求解器不包含耦合逻辑
 * - 材料系数由外部设置，求解器持有非拥有引用
 * - 边界条件接口具有物理意义
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// 初始化求解器（不设置材料系数，需单独调用 setMaterial）
    /// @param mesh 网格
    bool initialize(const Mesh& mesh);
    
    // =========================================================================
    // 边界条件接口（物理语义）
    // =========================================================================
    
    /// 添加固定位移边界条件
    /// @param boundaryId 边界ID
    /// @param displacement 位移向量 (m)
    void addFixedDisplacementBC(int boundaryId, const Vector3& displacement) {
        displacementBCs_[boundaryId] = displacement;
    }
    
    /// 批量设置固定位移边界条件
    void setFixedDisplacementBCs(const std::vector<std::pair<int, Vector3>>& bcs) {
        for (const auto& [id, val] : bcs) {
            displacementBCs_[id] = val;
        }
    }
    
    /// 添加固定分量边界条件
    /// @param boundaryId 边界ID
    /// @param component 分量索引 (0=x, 1=y, 2=z)
    /// @param value 位移值 (m)
    void addFixedComponentBC(int boundaryId, int component, Real value) {
        componentBCs_[boundaryId * 3 + component] = value;
    }
    
    /// 清除边界条件
    void clearBoundaryConditions() { 
        displacementBCs_.clear(); 
        componentBCs_.clear();
    }
    
    // =========================================================================
    // 材料系数接口
    // =========================================================================
    
    /// 设置材料系数（非拥有指针，生命周期由调用者管理）
    /// @param E 杨氏模量系数
    /// @param nu 泊松比系数
    void setMaterial(const Coefficient* E, const Coefficient* nu) {
        E_ = E;
        nu_ = nu;
    }
    
    /// 设置杨氏模量系数（非拥有指针）
    void setYoungModulus(const Coefficient* E) { E_ = E; }
    
    /// 设置泊松比系数（非拥有指针）
    void setPoissonRatio(const Coefficient* nu) { nu_ = nu; }
    
    // =========================================================================
    // 耦合载荷接口
    // =========================================================================
    
    /// 添加线性积分器（用于耦合载荷，如热膨胀）
    /// 注意：积分器由求解器持有所有权
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
    
    /// 获取应力场
    const GridFunction& stress() const { return *stress_; }
    
    /// 获取应变场
    const GridFunction& strain() const { return *strain_; }
    
private:
    void computeStressStrain();
    
    std::unique_ptr<GridFunction> u_;        ///< 位移场
    std::unique_ptr<GridFunction> stress_;   ///< 应力场 (6分量)
    std::unique_ptr<GridFunction> strain_;   ///< 应变场 (6分量)
    
    const Coefficient* E_ = nullptr;         ///< 杨氏模量（非拥有）
    const Coefficient* nu_ = nullptr;        ///< 泊松比（非拥有）
    
    std::vector<std::unique_ptr<VectorDomainLinearIntegrator>> linearIntegrators_;
    
    std::map<int, Vector3> displacementBCs_; ///< 边界ID -> 位移向量
    std::map<int, Real> componentBCs_;       ///< boundaryId*3+component -> value
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP
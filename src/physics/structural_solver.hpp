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
 * 设计原则：单场求解器，不包含任何耦合逻辑。
 * 热膨胀等耦合效果应通过外部设置体积力或修改边界条件实现。
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// 初始化求解器
    /// @param mesh 网格
    /// @param youngModulus 杨氏模量系数
    /// @param poissonRatio 泊松比系数
    bool initialize(const Mesh& mesh,
                    const Coefficient& youngModulus,
                    const Coefficient& poissonRatio);
    
    /// 添加 Dirichlet 边界条件
    /// @param boundaryId 边界ID
    /// @param disp 位移向量 (x, y, z)
    void addDirichletBC(int boundaryId, const Vector3& disp) {
        bcValues_[boundaryId] = disp;
    }
    
    /// 添加分量 Dirichlet 边界条件
    /// @param boundaryId 边界ID
    /// @param component 分量索引 (0=x, 1=y, 2=z)
    /// @param value 位移值
    void addDirichletBCComponent(int boundaryId, int component, Real value) {
        componentBCs_[boundaryId * 3 + component] = value;
    }
    
    /// 添加线性积分器（用于耦合载荷）
    void addLinearIntegrator(std::unique_ptr<VectorDomainLinearIntegrator> integrator) {
        linearIntegrators_.push_back(std::move(integrator));
    }
    
    /// 清除线性积分器
    void clearLinearIntegrators() { linearIntegrators_.clear(); }
    
    void clearBoundaryConditions() { 
        bcValues_.clear(); 
        componentBCs_.clear();
    }
    
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
    
    std::unique_ptr<GridFunction> u_;        // 位移
    std::unique_ptr<GridFunction> stress_;   // 应力 (6分量)
    std::unique_ptr<GridFunction> strain_;   // 应变 (6分量)
    
    const Coefficient* E_ = nullptr;         // 杨氏模量（非拥有）
    const Coefficient* nu_ = nullptr;        // 泊松比（非拥有）
    
    std::vector<std::unique_ptr<VectorDomainLinearIntegrator>> linearIntegrators_;
    
    std::map<int, Vector3> bcValues_;        // 边界ID -> 位移向量
    std::map<int, Real> componentBCs_;       // boundaryId*3+component -> value
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP

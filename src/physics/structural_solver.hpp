#ifndef MPFEM_STRUCTURAL_SOLVER_HPP
#define MPFEM_STRUCTURAL_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "fe/coefficient.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief 结构力学求解器（线弹性）
 * 
 * 求解: -div(sigma) = 0
 * 其中 sigma = C : (epsilon - epsilon_thermal)
 * epsilon_thermal = alpha_T * (T - T_ref) * I
 */
class StructuralSolver : public PhysicsFieldSolver {
public:
    StructuralSolver() = default;
    explicit StructuralSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::Displacement; }
    std::string fieldName() const override { return "Displacement"; }
    
    /// 使用单参数初始化（为了兼容基类接口）
    bool initialize(const Mesh& mesh, const PWConstCoefficient& param) override {
        // 默认使用param作为杨氏模量，泊松比设为0.3
        return initialize(mesh, param, PWConstCoefficient(std::vector<Real>{0.3}));
    }
    
    /// 完整初始化
    bool initialize(const Mesh& mesh,
                    const PWConstCoefficient& youngModulus,
                    const PWConstCoefficient& poissonRatio);
    
    void addDirichletBC(int boundaryId, Real value) override {
        bcValues_[boundaryId] = Vector3(value, value, value);
    }
    
    /// 添加向量Dirichlet边界条件
    void addDirichletBC(int boundaryId, const Vector3& disp) {
        bcValues_[boundaryId] = disp;
    }
    
    /// 添加分量Dirichlet边界条件 (0=x, 1=y, 2=z)
    void addDirichletBC(int boundaryId, int component, Real value) {
        componentBCs_[boundaryId * 3 + component] = value;
    }
    
    void clearBoundaryConditions() override { 
        bcValues_.clear(); 
        componentBCs_.clear();
    }
    
    /// 设置热应变系数: alpha_T * (T - T_ref)
    void setThermalStrain(const VectorCoefficient* thermalStrain) {
        thermalStrain_ = thermalStrain;
    }
    
    /// 设置温度场（非拥有）
    void setTemperatureField(const GridFunction* T) { T_ = T; }
    
    /// 设置参考温度
    void setReferenceTemperature(Real Tref) { Tref_ = Tref; }
    
    /// 设置热膨胀系数（非拥有）
    void setThermalExpansion(const Coefficient* alphaT) { alphaT_ = alphaT; }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *u_; }
    GridFunction& field() override { return *u_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    
    /// 获取应力场
    const GridFunction& stress() const { return *stress_; }
    
    /// 获取应变场
    const GridFunction& strain() const { return *strain_; }
    
private:
    void computeStressStrain();
    
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> u_;        // 位移
    std::unique_ptr<GridFunction> stress_;   // 应力 (6分量)
    std::unique_ptr<GridFunction> strain_;   // 应变 (6分量)
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient EInternal_;
    PWConstCoefficient nuInternal_;
    
    const GridFunction* T_ = nullptr;        // 温度场
    const Coefficient* alphaT_ = nullptr;    // 热膨胀系数
    const VectorCoefficient* thermalStrain_ = nullptr;
    Real Tref_ = 293.15;                     // 参考温度
    
    std::map<int, Vector3> bcValues_;
    std::map<int, Real> componentBCs_;
};

}  // namespace mpfem

#endif  // MPFEM_STRUCTURAL_SOLVER_HPP

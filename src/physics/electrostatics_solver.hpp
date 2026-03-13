#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include "assembly/assembler.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief 静电场求解器 - 最小化设计
 * 
 * 只持有必要的成员：FE空间、解、组装器
 * 边界条件通过assemble()参数传入
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    bool initialize(const Mesh& mesh, 
                    const PWConstCoefficient& conductivity) override;
    
    void addDirichletBC(int boundaryId, Real value) override {
        bcValues_[boundaryId] = value;
    }
    
    void clearBoundaryConditions() override { bcValues_.clear(); }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *V_; }
    GridFunction& field() override { return *V_; }
    const FESpace& feSpace() const override { return *fes_; }
    Index numDofs() const override { return fes_->numDofs(); }
    
    /// 设置电导率（非拥有指针）
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// 获取电导率
    const Coefficient* conductivity() const { return sigma_; }
    
    /// 计算焦耳热
    void computeJouleHeat(std::vector<Real>& Q) const;
    
private:
    // 最小成员集
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<GridFunction> V_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
    
    PWConstCoefficient sigmaInternal_;  // 内部拥有的电导率
    const Coefficient* sigma_ = nullptr;  // 当前使用的电导率（非拥有）
    
    std::map<int, Real> bcValues_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP

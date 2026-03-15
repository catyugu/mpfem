#ifndef MPFEM_ELECTROSTATICS_SOLVER_HPP
#define MPFEM_ELECTROSTATICS_SOLVER_HPP

#include "physics_field_solver.hpp"
#include <map>

namespace mpfem {

/**
 * @brief 静电场求解器
 * 
 * 求解：-div(sigma * grad V) = 0
 * 
 * 设计原则：单场求解器不包含耦合逻辑。
 * 温度依赖电导率应通过setConductivity()由外部设置。
 */
class ElectrostaticsSolver : public PhysicsFieldSolver {
public:
    ElectrostaticsSolver() = default;
    explicit ElectrostaticsSolver(int order) { order_ = order; }
    
    FieldKind fieldKind() const override { return FieldKind::ElectricPotential; }
    std::string fieldName() const override { return "ElectricPotential"; }
    
    /// 初始化求解器
    /// @param mesh 网格
    /// @param conductivity 电导率系数
    bool initialize(const Mesh& mesh, const Coefficient& conductivity);
    
    void addDirichletBC(int boundaryId, Real value) {
        bcValues_[boundaryId] = value;
    }
    
    void clearBoundaryConditions() { bcValues_.clear(); }
    
    void assemble() override;
    bool solve() override;
    
    const GridFunction& field() const override { return *V_; }
    GridFunction& field() override { return *V_; }
    
    /// 设置电导率系数（非拥有指针）
    void setConductivity(const Coefficient* sigma) { sigma_ = sigma; }
    
    /// 获取电导率系数
    const Coefficient* conductivity() const { return sigma_; }
    
private:
    std::unique_ptr<GridFunction> V_;
    const Coefficient* sigma_ = nullptr;
    std::map<int, Real> bcValues_;
};

} // namespace mpfem

#endif // MPFEM_ELECTROSTATICS_SOLVER_HPP
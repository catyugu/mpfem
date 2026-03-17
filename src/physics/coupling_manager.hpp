#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "model/field_kind.hpp"
#include "core/logger.hpp"
#include <memory>

namespace mpfem {

struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

/**
 * @brief 电-热-结构耦合管理器
 * 
 * 设计原则：
 * - 仅负责迭代控制逻辑和收敛判断
 * - 求解器引用为非拥有指针，生命周期由外部管理
 * - 耦合系数在初始化时通过求解器接口设置，系数绑定场引用后自动获取最新值
 */
class CouplingManager {
public:
    CouplingManager() = default;
    
    // =========================================================================
    // 求解器设置（非拥有指针）
    // =========================================================================
    
    void setElectrostaticsSolver(ElectrostaticsSolver* s) { esSolver_ = s; }
    void setHeatTransferSolver(HeatTransferSolver* s) { htSolver_ = s; }
    void setStructuralSolver(StructuralSolver* s) { stSolver_ = s; }
    

    
    // =========================================================================
    // 求解参数
    // =========================================================================
    
    void setTolerance(Real tol) { tol_ = tol; }
    void setMaxIterations(int n) { maxIter_ = n; }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    /// 执行耦合求解
    CouplingResult solve();

private:
    Real computeError();
    
    // 求解器引用（非拥有）
    ElectrostaticsSolver* esSolver_ = nullptr;
    HeatTransferSolver* htSolver_ = nullptr;
    StructuralSolver* stSolver_ = nullptr;
    
    // 迭代状态
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP

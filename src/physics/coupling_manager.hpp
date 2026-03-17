#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "fe/coefficient.hpp"
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
 * - 耦合逻辑集中在此，不在单场求解器中
 * - 求解器引用为非拥有指针，生命周期由外部管理
 * - 系数由调用者持有，注入场引用后自动获取最新值
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
    // 耦合系数设置（非拥有指针，由调用者持有）
    // =========================================================================
    
    /// 设置温度依赖电导率（非拥有，系数持有温度场引用）
    void setTemperatureDependentConductivity(TemperatureDependentConductivity* coef) {
        tempDepSigma_ = coef;
    }
    
    /// 设置焦耳热系数（非拥有，系数持有电势场和电导率引用）
    void setJouleHeatCoefficient(JouleHeatCoefficient* coef) {
        jouleHeat_ = coef;
    }
    
    /// 设置热膨胀系数（非拥有，系数持有温度场引用）
    void setThermalExpansionCoefficient(ThermalExpansionCoefficient* coef) {
        thermalExp_ = coef;
    }
    
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
    
    // 耦合系数引用（非拥有，由调用者持有）
    TemperatureDependentConductivity* tempDepSigma_ = nullptr;
    JouleHeatCoefficient* jouleHeat_ = nullptr;
    ThermalExpansionCoefficient* thermalExp_ = nullptr;
    
    // 迭代状态
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP

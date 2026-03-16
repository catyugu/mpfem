#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "coupling/joule_heating.hpp"
#include "fe/coefficient.hpp"
#include "core/logger.hpp"
#include <memory>
#include <set>
#include <map>

namespace mpfem {

enum class IterationMethod { Picard, Anderson };

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
 * - 耦合专用系数由本类持有所有权
 * 
 * 所有权策略：
 * - esSolver_, htSolver_, stSolver_: 非拥有指针，由 PhysicsProblemSetup 管理
 * - structE_, structNu_: 非拥有指针，由 PhysicsProblemSetup 管理
 * - jouleHeating_, tempDepSigma_, thermalAlphaCoef_: 拥有所有权
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
    // 温度依赖电导率设置
    // =========================================================================
    
    /// 启用温度依赖电导率
    void enableTempDependentConductivity(const std::set<int>& domains = {}) {
        tempDepDomains_ = domains;
        hasTempDepSigma_ = true;
    }
    
    /// 设置温度依赖材料参数
    void setTempDepMaterial(int domainId, Real rho0, Real alpha, Real tref);
    
    /// 设置常量电导率
    void setConstantConductivity(int domainId, Real sigma);
    
    // =========================================================================
    // 焦耳热耦合设置
    // =========================================================================
    
    /// 设置焦耳热作用域
    void setJouleHeatDomains(const std::set<int>& domains) {
        jouleHeatDomains_ = domains;
    }
    
    // =========================================================================
    // 热膨胀耦合设置
    // =========================================================================
    
    /// 设置热膨胀参数
    void setThermalExpansion(int domainId, Real alphaT, Real Tref) {
        thermalAlpha_[domainId] = alphaT;
        thermalTref_ = Tref;
    }
    
    /// 设置结构场材料参数（非拥有指针，用于热膨胀计算）
    void setStructuralMaterial(const Coefficient* E, const Coefficient* nu) {
        structE_ = E;
        structNu_ = nu;
    }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    /// 执行耦合求解
    CouplingResult solve();
    
private:
    void ensureTempDepSigma();
    void updateJouleHeat();
    void solveThermalExpansion();
    Real computeError();
    
    // 求解器引用（非拥有）
    ElectrostaticsSolver* esSolver_ = nullptr;
    HeatTransferSolver* htSolver_ = nullptr;
    StructuralSolver* stSolver_ = nullptr;
    
    // 耦合模块（拥有所有权）
    std::unique_ptr<JouleHeatingCoupling> jouleHeating_;
    std::unique_ptr<TemperatureDependentConductivity> tempDepSigma_;
    std::unique_ptr<PWConstCoefficient> thermalAlphaCoef_;
    
    // 外部材料参数（非拥有）
    const Coefficient* structE_ = nullptr;
    const Coefficient* structNu_ = nullptr;
    
    // 配置参数
    std::set<int> tempDepDomains_;
    std::set<int> jouleHeatDomains_;
    std::map<int, Real> thermalAlpha_;
    Real thermalTref_ = 293.15;
    bool hasTempDepSigma_ = false;
    
    // 迭代状态
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP

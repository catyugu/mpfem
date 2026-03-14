#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "coupling/joule_heating.hpp"
#include "fe/coefficient.hpp"
#include "core/logger.hpp"
#include <deque>
#include <memory>
#include <set>

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
 * 设计原则：耦合逻辑集中在此，不在单场求解器中。
 * 管理：
 * - 温度依赖电导率耦合
 * - 焦耳热耦合
 * - 热膨胀耦合
 */
class CouplingManager {
public:
    CouplingManager() = default;
    
    void setElectrostaticsSolver(ElectrostaticsSolver* s) { esSolver_ = s; }
    void setHeatTransferSolver(HeatTransferSolver* s) { htSolver_ = s; }
    void setStructuralSolver(StructuralSolver* s) { stSolver_ = s; }
    void setTolerance(Real tol) { tol_ = tol; }
    void setMaxIterations(int n) { maxIter_ = n; }
    
    /// 启用温度依赖电导率
    void enableTempDependentConductivity(const std::set<int>& domains = {}) {
        tempDepDomains_ = domains;
        hasTempDepSigma_ = true;
    }
    
    /// 设置焦耳热作用域
    void setJouleHeatDomains(const std::set<int>& domains) {
        jouleHeatDomains_ = domains;
    }
    
    /// 设置温度依赖材料参数
    void setTempDepMaterial(int domainId, Real rho0, Real alpha, Real tref) {
        ensureTempDepSigma();
        tempDepSigma_->setMaterial(domainId, rho0, alpha, tref);
    }
    
    /// 设置常量电导率
    void setConstantConductivity(int domainId, Real sigma) {
        ensureTempDepSigma();
        tempDepSigma_->setConstantConductivity(domainId, sigma);
    }
    
    /// 设置热膨胀参数
    void setThermalExpansion(int domainId, Real alphaT, Real Tref) {
        thermalAlpha_[domainId] = alphaT;
        thermalTref_ = Tref;
    }
    
    /// 执行耦合求解
    CouplingResult solve() {
        CouplingResult result;
        if (!esSolver_ || !htSolver_) return result;
        
        for (int i = 0; i < maxIter_; ++i) {
            // 更新温度依赖电导率
            if (hasTempDepSigma_ && tempDepSigma_) {
                tempDepSigma_->setTemperatureField(&htSolver_->field());
                esSolver_->setConductivity(tempDepSigma_.get());
            }
            
            // 求解电场
            esSolver_->assemble();
            esSolver_->solve();
            
            // 更新焦耳热并求解热场
            updateJouleHeat();
            htSolver_->assemble();
            htSolver_->solve();
            
            // 计算误差
            Real err = computeError();
            result.iterations = i + 1;
            result.residual = err;
            
            LOG_INFO << "Coupling iteration " << (i+1) << ", residual = " << err;
            
            if (err < tol_) {
                result.converged = true;
                break;
            }
        }
        
        // 后处理：求解结构场（如果有热膨胀耦合）
        if (stSolver_ && !thermalAlpha_.empty()) {
            solveThermalExpansion();
        }
        
        return result;
    }
    
private:
    void ensureTempDepSigma() {
        if (!tempDepSigma_) {
            tempDepSigma_ = std::make_unique<TemperatureDependentConductivity>();
        }
    }
    
    void updateJouleHeat() {
        if (!jouleHeating_) {
            jouleHeating_ = std::make_unique<JouleHeatingCoupling>();
            jouleHeating_->setDomains(jouleHeatDomains_);
        }
        
        jouleHeating_->setPotentialField(&esSolver_->field());
        jouleHeating_->setConductivity(esSolver_->conductivity());
        htSolver_->setHeatSource(jouleHeating_->getHeatSource());
    }
    
    void solveThermalExpansion() {
        // 获取最大域ID
        int maxDomainId = 0;
        for (const auto& [domId, _] : thermalAlpha_) {
            maxDomainId = std::max(maxDomainId, domId);
        }
        
        // 创建热膨胀系数
        auto alphaCoef = std::make_unique<PWConstCoefficient>(maxDomainId);
        for (const auto& [domId, alpha] : thermalAlpha_) {
            alphaCoef->set(domId, alpha);
        }
        
        // 设置热膨胀参数
        stSolver_->setTemperatureField(&htSolver_->field());
        stSolver_->setReferenceTemperature(thermalTref_);
        stSolver_->setThermalExpansion(alphaCoef.get());
        
        // 组装并求解
        stSolver_->assemble();
        stSolver_->solve();
    }
    
    Real computeError() {
        if (prevT_.size() == 0) {
            prevT_ = htSolver_->field().values();
            return 1.0;
        }
        Real diff = (htSolver_->field().values() - prevT_).norm();
        prevT_ = htSolver_->field().values();
        return diff / (htSolver_->field().values().norm() + 1e-15);
    }
    
    ElectrostaticsSolver* esSolver_ = nullptr;
    HeatTransferSolver* htSolver_ = nullptr;
    StructuralSolver* stSolver_ = nullptr;
    
    // 耦合模块
    std::unique_ptr<JouleHeatingCoupling> jouleHeating_;
    std::unique_ptr<TemperatureDependentConductivity> tempDepSigma_;
    
    // 配置
    std::set<int> tempDepDomains_;
    std::set<int> jouleHeatDomains_;
    std::map<int, Real> thermalAlpha_;
    Real thermalTref_ = 293.15;
    bool hasTempDepSigma_ = false;
    
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP

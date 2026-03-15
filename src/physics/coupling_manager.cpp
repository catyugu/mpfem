#include "coupling_manager.hpp"

namespace mpfem {

void CouplingManager::setTempDepMaterial(int domainId, Real rho0, Real alpha, Real tref) {
    ensureTempDepSigma();
    tempDepSigma_->setMaterial(domainId, rho0, alpha, tref);
}

void CouplingManager::setConstantConductivity(int domainId, Real sigma) {
    ensureTempDepSigma();
    tempDepSigma_->setConstantConductivity(domainId, sigma);
}

void CouplingManager::ensureTempDepSigma() {
    if (!tempDepSigma_) {
        tempDepSigma_ = std::make_unique<TemperatureDependentConductivity>();
    }
}

void CouplingManager::updateJouleHeat() {
    if (!jouleHeating_) {
        jouleHeating_ = std::make_unique<JouleHeatingCoupling>();
        jouleHeating_->setDomains(jouleHeatDomains_);
    }
    
    jouleHeating_->setPotentialField(&esSolver_->field());
    jouleHeating_->setConductivity(esSolver_->conductivity());
    htSolver_->setHeatSource(jouleHeating_->getHeatSource());
}

void CouplingManager::solveThermalExpansion() {
    // 获取最大域ID
    int maxDomainId = 0;
    for (const auto& [domId, _] : thermalAlpha_) {
        maxDomainId = std::max(maxDomainId, domId);
    }
    
    // 创建热膨胀系数
    thermalAlphaCoef_ = std::make_unique<PWConstCoefficient>(maxDomainId);
    for (const auto& [domId, alpha] : thermalAlpha_) {
        thermalAlphaCoef_->set(domId, alpha);
    }
    
    // 添加热膨胀载荷积分器到结构求解器
    auto thermalLoad = std::make_unique<ThermalLoadIntegrator>(
        structE_, structNu_, thermalAlphaCoef_.get(), 
        &htSolver_->field(), thermalTref_);
    stSolver_->addLinearIntegrator(std::move(thermalLoad));
    
    // 组装并求解
    stSolver_->assemble();
    stSolver_->solve();
}

Real CouplingManager::computeError() {
    if (prevT_.size() == 0) {
        prevT_ = htSolver_->field().values();
        return 1.0;
    }
    Real diff = (htSolver_->field().values() - prevT_).norm();
    prevT_ = htSolver_->field().values();
    return diff / (htSolver_->field().values().norm() + 1e-15);
}

CouplingResult CouplingManager::solve() {
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

}  // namespace mpfem

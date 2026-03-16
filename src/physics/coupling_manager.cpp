#include "coupling_manager.hpp"
#include "core/logger.hpp"

namespace mpfem {

const GridFunction* CouplingManager::getField(const std::string& name) const {
    if (name == "V" || name == "electric_potential") {
        return esSolver_ ? &esSolver_->field() : nullptr;
    }
    else if (name == "T" || name == "temperature") {
        return htSolver_ ? &htSolver_->field() : nullptr;
    }
    else if (name == "u" || name == "displacement") {
        return stSolver_ ? &stSolver_->field() : nullptr;
    }
    return nullptr;
}

void CouplingManager::setCoefficient(const std::string& name, 
                                      const std::set<int>& domains,
                                      const Coefficient* coef) {
    coefficients_[name].set(domains, coef);
}

void CouplingManager::setCoefficient(const std::string& name, const Coefficient* coef) {
    coefficients_[name].setAll(coef);
}

const Coefficient* CouplingManager::getCoefficient(const std::string& name) const {
    auto it = ownedCoefficients_.find(name);
    if (it != ownedCoefficients_.end()) {
        return it->second.get();
    }
    return nullptr;
}

Real CouplingManager::computeError() {
    if (!htSolver_) return 0.0;
    
    if (prevT_.size() == 0) {
        prevT_ = htSolver_->field().values();
        return 1.0;
    }
    Real diff = (htSolver_->field().values() - prevT_).norm();
    prevT_ = htSolver_->field().values();
    return diff / (htSolver_->field().values().norm() + 1e-15);
}

CouplingResult CouplingManager::solve() {
    ScopedTimer timer("Coupling solve");
    
    CouplingResult result;
    if (!esSolver_ || !htSolver_) return result;
    
    // 检查是否有温度依赖电导率
    bool hasTempDepSigma = ownedCoefficients_.find("temp_dep_sigma") != ownedCoefficients_.end();
    
    // 检查是否有焦耳热系数
    bool hasJouleHeat = ownedCoefficients_.find("joule_heat") != ownedCoefficients_.end();
    
    for (int i = 0; i < maxIter_; ++i) {
        // 更新温度依赖电导率
        if (hasTempDepSigma) {
            auto* tempDepSigma = dynamic_cast<TemperatureDependentConductivity*>(
                ownedCoefficients_["temp_dep_sigma"].get());
            if (tempDepSigma) {
                tempDepSigma->setTemperatureField(&htSolver_->field());
                esSolver_->setConductivity(tempDepSigma);
            }
        }
        
        // 求解电场
        esSolver_->assemble();
        esSolver_->solve();
        
        // 更新焦耳热并求解热场
        if (hasJouleHeat) {
            auto* jouleHeat = dynamic_cast<JouleHeatCoefficient*>(
                ownedCoefficients_["joule_heat"].get());
            if (jouleHeat) {
                jouleHeat->setPotential(&esSolver_->field());
                jouleHeat->setConductivity(&esSolver_->conductivity());
                htSolver_->setHeatSource(jouleHeat);
            }
        }
        
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
    if (stSolver_ && ownedCoefficients_.find("thermal_expansion") != ownedCoefficients_.end()) {
        auto* thermalExp = dynamic_cast<ThermalExpansionCoefficient*>(
            ownedCoefficients_["thermal_expansion"].get());
        
        if (thermalExp) {
            thermalExp->setTemperatureField(&htSolver_->field());
            
            // 添加热膨胀载荷积分器
            auto thermalLoad = std::make_unique<ThermalLoadIntegrator>(
                &stSolver_->youngModulus(), &stSolver_->poissonRatio(), 
                thermalExp, &htSolver_->field(), 293.15);
            stSolver_->addLinearIntegrator(std::move(thermalLoad));
            
            // 组装并求解
            stSolver_->assemble();
            stSolver_->solve();
        }
    }
    
    return result;
}

}  // namespace mpfem

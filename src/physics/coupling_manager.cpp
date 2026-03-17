#include "coupling_manager.hpp"
#include "core/logger.hpp"

namespace mpfem {

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
    
    for (int i = 0; i < maxIter_; ++i) {
        // 温度依赖电导率：系数持有温度场引用，自动获取最新值
        if (tempDepSigma_) {
            esSolver_->setConductivity(tempDepSigma_);
        }
        
        // 求解电场
        esSolver_->assemble();
        esSolver_->solve();
        
        // 焦耳热：系数持有电势场和电导率引用，自动获取最新值
        if (jouleHeat_) {
            htSolver_->setHeatSource(jouleHeat_);
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
    
    // 后处理：求解结构场（热膨胀耦合）
    if (stSolver_ && thermalExp_) {
        // 热膨胀系数持有温度场引用，自动获取最新值
        auto thermalLoad = std::make_unique<ThermalLoadIntegrator>(
            &stSolver_->youngModulus(), &stSolver_->poissonRatio(), 
            thermalExp_);
        stSolver_->addLinearIntegrator(std::move(thermalLoad));
        
        stSolver_->assemble();
        stSolver_->solve();
    }
    
    return result;
}

}  // namespace mpfem

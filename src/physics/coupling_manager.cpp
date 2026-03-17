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
        // 求解电场
        // 温度依赖电导率已在初始化时设置，系数绑定温度场引用自动获取最新值
        esSolver_->assemble();
        esSolver_->solve();
        
        // 求解温度场
        // 焦耳热系数已在初始化时设置，绑定电势场引用自动获取最新值
        htSolver_->assemble();
        htSolver_->solve();
        
        // 计算收敛误差
        Real err = computeError();
        result.iterations = i + 1;
        result.residual = err;
        
        LOG_INFO << "Coupling iteration " << (i+1) << ", residual = " << err;
        
        if (err < tol_) {
            result.converged = true;
            break;
        }
    }
    
    // 后处理：求解结构场
    // 热膨胀积分器已在初始化时添加，系数绑定温度场引用自动获取最新值
    if (stSolver_) {
        stSolver_->assemble();
        stSolver_->solve();
    }
    
    return result;
}

}  // namespace mpfem

#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "core/logger.hpp"
#include <deque>

namespace mpfem {

enum class IterationMethod { Picard, Anderson };

struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

class CouplingManager {
public:
    CouplingManager() = default;
    
    void setElectrostaticsSolver(ElectrostaticsSolver* s) { esSolver_ = s; }
    void setHeatTransferSolver(HeatTransferSolver* s) { htSolver_ = s; }
    void setTolerance(Real tol) { tol_ = tol; }
    void setMaxIterations(int n) { maxIter_ = n; }
    
    CouplingResult solve() {
        CouplingResult result;
        if (!esSolver_ || !htSolver_) return result;
        
        for (int i = 0; i < maxIter_; ++i) {
            // 解静电场
            esSolver_->assemble();
            esSolver_->solve();
            
            // 更新焦耳热并解热传导
            updateJouleHeat();
            htSolver_->assemble();
            htSolver_->solve();
            
            // 计算误差
            Real err = computeError();
            result.iterations = i + 1;
            result.residual = err;
            
            if (err < tol_) {
                result.converged = true;
                break;
            }
        }
        return result;
    }
    
private:
    void updateJouleHeat() {
        if (!jouleHeat_) {
            jouleHeat_ = std::make_unique<JouleHeatCoefficient>();
            htSolver_->setHeatSource(jouleHeat_.get());
        }
        
        // 设置焦耳热回调
        auto* V = &esSolver_->field();
        const auto* sigma = esSolver_->conductivity();
        
        jouleHeat_->setGradientFunc([V](int e, const Real* xi, ElementTransform& t) {
            return V->gradient(e, xi, t);
        });
        jouleHeat_->setConductivityFunc([sigma](ElementTransform& t) {
            return sigma ? sigma->eval(t) : 1.0;
        });
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
    std::unique_ptr<JouleHeatCoefficient> jouleHeat_;
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP
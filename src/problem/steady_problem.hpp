#ifndef MPFEM_STEADY_PROBLEM_HPP
#define MPFEM_STEADY_PROBLEM_HPP

#include "problem/problem.hpp"
#include "problem/transient_problem.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "core/logger.hpp"

namespace mpfem {

/**
 * @brief 耦合求解结果
 */
struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

/**
 * @brief 稳态问题
 * 
 * 继承Problem数据基类，添加求解器所有权和求解方法。
 */
class SteadyProblem : public Problem {
public:
    // 求解器（拥有所有权）
    std::unique_ptr<ElectrostaticsSolver> electrostatics;
    std::unique_ptr<HeatTransferSolver> heatTransfer;
    std::unique_ptr<StructuralSolver> structural;
    
    // 耦合参数
    int couplingMaxIter = 15;
    Real couplingTol = 1e-6;
    
    // 查询方法
    bool hasElectrostatics() const { return electrostatics != nullptr; }
    bool hasHeatTransfer() const { return heatTransfer != nullptr; }
    bool hasStructural() const { return structural != nullptr; }
    bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
    bool hasThermalExpansion() const { return hasHeatTransfer() && hasStructural(); }
    bool isCoupled() const { return hasJouleHeating() || hasThermalExpansion(); }
    
    /**
     * @brief 执行耦合或单场求解
     */
    CouplingResult solve() {
        ScopedTimer timer("Coupling solve");
        CouplingResult result;

        if (!isCoupled()) {
            if (hasElectrostatics()) {
                electrostatics->assemble();
                electrostatics->solve();
            }
            return result;
        }

        for (int i = 0; i < couplingMaxIter; ++i) {
            electrostatics->assemble();
            electrostatics->solve();
            heatTransfer->assemble();
            heatTransfer->solve();
            
            Real err = computeCouplingError();
            result.iterations = i + 1;
            result.residual = err;
            LOG_INFO << "Coupling iteration " << (i + 1) << ", residual = " << err;
            if (err < couplingTol) { result.converged = true; break; }
        }

        if (hasStructural()) {
            structural->assemble();
            structural->solve();
        }
        return result;
    }

private:
    Vector prevT_;
    
    Real computeCouplingError() {
        if (!heatTransfer) return 0.0;
        const auto& T = heatTransfer->field().values();
        if (prevT_.size() == 0) { prevT_ = T; return 1.0; }
        Real diff = (T - prevT_).norm();
        prevT_ = T;
        return diff / (T.norm() + 1e-15);
    }
};

}  // namespace mpfem

#endif  // MPFEM_STEADY_PROBLEM_HPP

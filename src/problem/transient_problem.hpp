#ifndef MPFEM_TRANSIENT_PROBLEM_HPP
#define MPFEM_TRANSIENT_PROBLEM_HPP

#include "problem.hpp"
#include <deque>

namespace mpfem {

/**
 * @brief 时间积分方案
 */
enum class TimeScheme {
    BackwardEuler,   ///< 后向欧拉（一阶精度，无条件稳定）
    BDF2,            ///< 二阶后向差分（二阶精度）
    CrankNicolson    ///< Crank-Nicolson（二阶精度，无条件稳定）
};

/**
 * @brief 瞬态问题（数据部分）
 * 
 * 扩展Problem基类，添加时间维度数据。
 */
class TransientProblem : public Problem {
public:
    bool isTransient() const override { return true; }
    
    // 时间参数
    Real startTime = 0.0;
    Real endTime = 1.0;
    Real timeStep = 0.01;
    Real currentTime = 0.0;
    int currentStep = 0;
    
    /// 时间积分方案
    TimeScheme scheme = TimeScheme::BackwardEuler;
    
    /// 历史场存储（多步格式）
    std::map<std::string, std::deque<Vector>> historyFields;
    
    /// 前进一个时间步
    void advanceTime() {
        currentTime += timeStep;
        ++currentStep;
    }
    
    /// 是否到达终止时间
    bool finished() const {
        return currentTime >= endTime - 1e-10;
    }
};

}  // namespace mpfem

#endif  // MPFEM_TRANSIENT_PROBLEM_HPP
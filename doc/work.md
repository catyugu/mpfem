# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

识别并重构这个代码是一个非常经典的架构优化过程。当前代码确实存在明显的架构缺陷，导致难以支持自适应步长。

### 1. 现有设计反模式与问题分析

1. **上帝对象 (God Object) 与职责混淆**：`TransientProblem` 既是“数据容器”（持有哪些求解器、求解参数），又充当了“控制器”（实现了硬编码的 `while` 循环和 Picard 耦合迭代）。这违反了**单一职责原则 (SRP)**。
2. **僵化设计 (Rigidity)**：时间步进逻辑被“钉死”在 `TransientProblem::solve` 中，要想加入自适应步长，原方法会变得极其臃肿，且无法实现策略的灵活替换（违反**开闭原则 OCP**）。
3. **隐式的时间推进假设**：
   - 现有的 `FieldValues::advanceTime()` 是一种“推进后不可逆”的操作。如果自适应步长在某一步发散，缺乏状态回滚机制。
   - `BDF2Integrator` 硬编码了常数步长的系数（1.5, 2.0, -0.5），在变步长下这些系数是错误的，这是一种**隐蔽的数学缺陷**。
   - 快照（Snapshot）输出与时间步进强绑定，导致无法在“内部小步长”和“外部采样点”之间解耦。

---

### 2. 步骤化重构方案

我们的目标是：完全剥离 `TransientProblem` 中的控制流，引入独立的 `TimeStepper` 策略接口；解耦“计算步长”与“采样步长”；并修正变步长下所需的底层支持。

#### 第 1 步：定义 `TimeStepper` 接口，实现责任分离
将控制流从 `TransientProblem` 中抽离。

```cpp
// file: time/time_stepper.hpp
#ifndef MPFEM_TIME_STEPPER_HPP
#define MPFEM_TIME_STEPPER_HPP

#include "problem/transient_problem.hpp"
#include "time/time_integrator.hpp"

namespace mpfem {

class TimeStepper {
public:
    virtual ~TimeStepper() = default;

    // 接收问题实例并接管整个瞬态求解循环
    virtual TransientResult solve(TransientProblem& problem, TimeIntegrator& integrator) = 0;
};

} // namespace mpfem
#endif // MPFEM_TIME_STEPPER_HPP
```

#### 第 2 步：完善底层支持 (变步长 BDF 积分器 & 状态管理)
自适应步长必须处理 `dt` 的动态变化，现有的 BDF2 是常步长公式，必须升级为变步长公式，同时引入 `TimeHistory` 记录历史步长。

```cpp
// 修改 time_integrator.hpp
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;
    
    // 增加 prev_dt 参数支持变步长
    virtual bool step(PhysicsFieldSolver& solver, FieldValues& history, 
                      Real dt, Real prev_dt, int currentStep) = 0;
};

// 修改 time/bdf2_integrator.cpp
bool BDF2Integrator::step(PhysicsFieldSolver& solver, FieldValues& history, 
                          Real dt, Real prev_dt, int currentStep) {
    if (currentStep == 0 || prev_dt <= 0.0) {
        // 第一步退化为 BDF1
        const Vector historyCombo = history.history(solver.fieldName(), 1).values();
        return solver.solveTransientStep(1.0, dt, dt, historyCombo);
    }
    
    // 变步长 BDF2 系数计算 (omega = dt / prev_dt)
    Real omega = dt / prev_dt;
    Real alpha = (1.0 + 2.0 * omega) / (1.0 + omega);
    Real beta_n = (1.0 + omega);
    Real beta_nm1 = -(omega * omega) / (1.0 + omega);
    
    const GridFunction& prev1 = history.history(solver.fieldName(), 1);
    const GridFunction& prev2 = history.history(solver.fieldName(), 2);
    
    Vector historyCombo = beta_n * prev1.values() + beta_nm1 * prev2.values();
    
    return solver.solveTransientStep(alpha, dt, dt, historyCombo);
}
```

#### 第 3 步：实现自适应步长策略 (AdaptiveTimeStepper)
实现核心要求：初始步长为采样的 1/5，最大步长为 10s，并且保证精确踩在采样点上以输出正确的值。

```cpp
// file: time/adaptive_time_stepper.hpp
#include "time/time_stepper.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

class AdaptiveTimeStepper : public TimeStepper {
public:
    AdaptiveTimeStepper(Real sampleStep, Real maxDt = 10.0) 
        : sampleStep_(sampleStep), maxDt_(maxDt) {}

    TransientResult solve(TransientProblem& problem, TimeIntegrator& integrator) override {
        TransientResult result;
        Real currentTime = problem.startTime;
        
        // 按照需求：初始步长为指定采样步长的 1/5
        Real dt = sampleStep_ / 5.0; 
        Real prev_dt = dt;
        int stepCount = 0;
        
        problem.initializeSteadyState();
        problem.fieldValues.advanceTime();
        result.addSnapshot(currentTime, problem.fieldValues);

        Real nextSampleTime = currentTime + sampleStep_;

        while (currentTime < problem.endTime - 1e-12) {
            // 保证精确落在采样点上
            bool isSamplingStep = false;
            if (currentTime + dt >= nextSampleTime - 1e-12) {
                dt = nextSampleTime - currentTime;
                isSamplingStep = true;
            }

            LOG_INFO << "Attempting step " << stepCount + 1 << ", t=" << currentTime + dt << ", dt=" << dt;

            // 耦合迭代
            Real errT = 0.0;
            bool converged = tryCouplingStep(problem, integrator, dt, prev_dt, errT);

            if (converged) {
                // 成功推进
                if (problem.hasStructural()) {
                    problem.structural->solveSteady();
                }
                
                currentTime += dt;
                prev_dt = dt;
                stepCount++;
                problem.fieldValues.advanceTime(); // 确认步进，将结果推入历史栈

                // 采样点输出
                if (isSamplingStep) {
                    result.addSnapshot(currentTime, problem.fieldValues);
                    nextSampleTime += sampleStep_;
                }

                // 步长自适应策略：收敛良好则放大步长
                dt = std::min(dt * 1.2, maxDt_); 
            } else {
                // 步进失败，无需调用 advanceTime()，状态自动回滚
                LOG_WARN << "Convergence failed at dt=" << dt << ", shrinking time step.";
                dt *= 0.5; 
                if (dt < 1e-6) {
                    LOG_ERROR << "Time step too small. Aborting.";
                    break;
                }
            }
        }
        
        result.timeSteps = stepCount;
        result.finalTime = currentTime;
        result.converged = (currentTime >= problem.endTime - 1e-12);
        return result;
    }

private:
    bool tryCouplingStep(TransientProblem& problem, TimeIntegrator& integrator, 
                         Real dt, Real prev_dt, Real& residual) {
        const bool hasE = problem.hasElectrostatics();
        const bool hasT = problem.hasHeatTransfer();
        
        if (!hasT) {
            if (hasE) problem.electrostatics->solveSteady();
            return true;
        }

        Vector prevT;
        for (int iter = 0; iter < problem.couplingMaxIter; ++iter) {
            if (hasE) problem.electrostatics->solveSteady();
            
            if (!integrator.step(*problem.heatTransfer, problem.fieldValues, dt, prev_dt, 1)) {
                return false;
            }
            
            const auto& currentT = problem.heatTransfer->field().values();
            if (iter == 0) { prevT = currentT; continue; }
            
            residual = (currentT - prevT).norm() / (currentT.norm() + 1e-15);
            prevT = currentT;
            
            if (residual < problem.couplingTol) return true;
        }
        return false;
    }

    Real sampleStep_;
    Real maxDt_;
};

} // namespace mpfem
```

#### 第 4 步：清理 `TransientProblem` (瘦身)
将 `TransientProblem` 简化为纯粹的模型表达，它不再包含 `while` 循环，并将 `solve` 方法委托给接口。这样我们在外面就可以轻松换掉 `Stepper`。

```cpp
// 修改 transient_problem.cpp
TransientResult TransientProblem::solve() {
    ScopedTimer timer("Transient solve");
    
    int requiredHistoryDepth = (scheme == TimeScheme::BDF2) ? 3 : 2;
    if (fieldValues.maxHistorySteps() < requiredHistoryDepth) {
        initializeTransient(requiredHistoryDepth);
    }
    
    auto integrator = createTimeIntegrator(scheme);
    
    // 使用注入的自适应步长策略替换原本僵化的 while 循环
    // 默认行为：指定的 timeStep 作为采样步长，最大限制为 10s
    AdaptiveTimeStepper stepper(timeStep, 10.0);
    
    return stepper.solve(*this, *integrator);
}
```

### 总结
这套重构方案做了以下几件事：
1. **控制反转**：剥离出了 `TimeStepper` 接口，解耦了业务逻辑。
2. **逻辑安全**：在重试阶段利用 `FieldValues` 不调用 `advanceTime` 即可实现天然的隐式**状态回滚**。
3. **数学严谨**：修正了 `BDF2` 算法以支持变步长 (`omega = dt/prev_dt`)，防止在自适应过程中产生数值上的虚假解。
4. **精准采样**：通过 `isSamplingStep` 判断和余量切割(`nextSampleTime - currentTime`)，保证了哪怕内部以 0.1s 或者 10s 在疯跑，到了 5s 整数倍时一定能切出一个刚好踩在这个时间点的帧并输出。不需要修改结果后处理代码，完全无痛。
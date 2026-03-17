# 大纲

依据@prompt.md ，适当参考学习external/mfem中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast, friend, mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 完成一块工作任务后：
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。
  * 提交一次代码，然后继续下一块工作。

## 工作任务1

* 目前代码有很多深嵌套，而且存在所有权管理不统一、不清晰的问题。
* CouplingManager的以下部分是反模式，破坏了系数分配系统的一致性，应该移除：

```cpp
    // =========================================================================
    // 耦合系数设置（非拥有指针，由调用者持有）
    // =========================================================================
    
    /// 设置温度依赖电导率（非拥有，系数持有温度场引用）
    void setTemperatureDependentConductivity(TemperatureDependentConductivity* coef) {
        tempDepSigma_ = coef;
    }
    
    /// 设置焦耳热系数（非拥有，系数持有电势场和电导率引用）
    void setJouleHeatCoefficient(JouleHeatCoefficient* coef) {
        jouleHeat_ = coef;
    }
    
    /// 设置热膨胀系数（非拥有，系数持有温度场引用）
    void setThermalExpansionCoefficient(ThermalExpansionCoefficient* coef) {
        thermalExp_ = coef;
    }

    // =========================================================================
    // ...
    // =========================================================================
    
    TemperatureDependentConductivity* tempDepSigma_ = nullptr;
    JouleHeatCoefficient* jouleHeat_ = nullptr;
    ThermalExpansionCoefficient* thermalExp_ = nullptr;
```

* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码。
* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。

```
 HUAWEI    mpfem  dev ≡  ~1 |  ~1  1   1.9s⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                                        
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [0ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [175ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [253ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [256ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [257ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [258ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [259ms] Building electrostatics solver, order = 2
[INFO] [420ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [420ms] Building heat transfer solver, order = 2
[INFO] [587ms] HeatTransferSolver: 49889 DOFs
[INFO] [587ms] Building structural solver, order = 2
[INFO] [1.99s] StructuralSolver: 149667 DOFs
[INFO] [1.99s] Joule heating domains: 7 domains
[INFO] [1.99s] Thermal expansion coupling enabled
[INFO] [1.99s] Running coupled electro-thermal solve...
[INFO] [2.09s] Electrostatics assemble completed in 0.093s
[INFO] [3.72s] [UMFPACK] Solve successful, solution norm: 1.61365
[INFO] [3.72s] Linear solve (UMFPACK) completed in 1.634s
[INFO] [3.72s] Electrostatics converged: iter=1 res=0
[INFO] [3.91s] HeatTransfer assemble completed in 0.183s
[INFO] [5.51s] [UMFPACK] Solve successful, solution norm: 72119.9
[INFO] [5.51s] Linear solve (UMFPACK) completed in 1.605s
[INFO] [5.51s] HeatTransfer converged: iter=1 res=0
[INFO] [5.51s] Coupling iteration 1, residual = 1
[INFO] [5.61s] Electrostatics assemble completed in 0.093s
[INFO] [5.68s] [UMFPACK] Solve successful, solution norm: 1.62178
[INFO] [5.68s] Linear solve (UMFPACK) completed in 0.073s
[INFO] [5.68s] Electrostatics converged: iter=1 res=0
[INFO] [5.86s] HeatTransfer assemble completed in 0.181s
[INFO] [5.92s] [UMFPACK] Solve successful, solution norm: 72061.2
[INFO] [5.92s] Linear solve (UMFPACK) completed in 0.058s
[INFO] [5.92s] HeatTransfer converged: iter=1 res=0
[INFO] [5.93s] Coupling iteration 2, residual = 0.00081761
[INFO] [6.01s] Electrostatics assemble completed in 0.085s
[INFO] [6.10s] [UMFPACK] Solve successful, solution norm: 1.62171
[INFO] [6.10s] Linear solve (UMFPACK) completed in 0.091s
[INFO] [6.10s] Electrostatics converged: iter=1 res=0
[INFO] [6.27s] HeatTransfer assemble completed in 0.166s
[INFO] [6.36s] [UMFPACK] Solve successful, solution norm: 72061.7
[INFO] [6.36s] Linear solve (UMFPACK) completed in 0.088s
[INFO] [6.36s] HeatTransfer converged: iter=1 res=0
[INFO] [6.36s] Coupling iteration 3, residual = 7.07832e-06
[INFO] [6.87s] Structural assemble completed in 0.503s
[INFO] [14.99s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [14.99s] Linear solve (UMFPACK) completed in 8.127s
[INFO] [14.99s] StructuralSolver: displacement norm = 0.00592757
[INFO] [14.99s] Coupling solve completed in 12.998s
[INFO] [14.99s] Coupling converged in 3 iterations
[INFO] [14.99s] Potential range: [0, 0.02] V
[INFO] [14.99s] Temperature range: [322.185, 330.042] K
[INFO] [14.99s] Temperature range: [49.0351, 56.8922] C
[INFO] [14.99s] Max displacement magnitude: 5.0786e-05 m
[INFO] [15.02s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [15.12s] Exported VTU results to results/busbar_results.vtu
[INFO] [15.12s] Exported VTU results to: results/busbar_results.vtu
[INFO] [15.18s] Exported results to results/mpfem_result.txt
[INFO] [15.18s] Exported results to: results/mpfem_result.txt
[INFO] [15.18s] === Example completed successfully! ===
```
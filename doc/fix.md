# 大纲

依据@prompt.md ，适当参考学习external/mfem和external/hpc-fem-playground中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
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
* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码。
* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。

```
 HUAWEI    mpfem  dev ≡  ~5   1.973s⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                    pwsh   97  22:41:02 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [1ms] Case directory: .\cases\busbar_order2
[INFO] [1ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [2ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [157ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [219ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [220ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [220ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [221ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [221ms] Building electrostatics solver, order = 2
[INFO] [386ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [386ms] Building heat transfer solver, order = 2
[INFO] [543ms] HeatTransferSolver: 49889 DOFs
[INFO] [543ms] Building structural solver, order = 2
[INFO] [1.94s] StructuralSolver: 149667 DOFs
[INFO] [1.94s] Joule heating domains: 7 domains
[INFO] [1.94s] Thermal expansion coupling enabled
[INFO] [1.94s] Running coupled electro-thermal solve...
[INFO] [2.02s] Electrostatics assemble completed in 0.083s
[INFO] [3.01s] Linear solve (CG+IC) completed in 0.986s
[INFO] [3.01s] Electrostatics converged: iter=250 res=8.58922e-11
[INFO] [3.34s] HeatTransfer assemble completed in 0.331s
[INFO] [4.28s] Linear solve (CG+IC) completed in 0.944s
[INFO] [4.28s] HeatTransfer converged: iter=263 res=8.72254e-11
[INFO] [4.28s] Coupling iteration 1, residual = 1
[INFO] [4.36s] Electrostatics assemble completed in 0.075s
[INFO] [5.35s] Linear solve (CG+IC) completed in 0.989s
[INFO] [5.35s] Electrostatics converged: iter=236 res=9.40067e-11
[INFO] [5.63s] HeatTransfer assemble completed in 0.282s
[INFO] [6.56s] Linear solve (CG+IC) completed in 0.932s
[INFO] [6.57s] HeatTransfer converged: iter=245 res=9.60864e-11
[INFO] [6.57s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.65s] Electrostatics assemble completed in 0.084s
[INFO] [7.41s] Linear solve (CG+IC) completed in 0.755s
[INFO] [7.41s] Electrostatics converged: iter=209 res=8.02274e-11
[INFO] [7.69s] HeatTransfer assemble completed in 0.279s
[INFO] [8.53s] Linear solve (CG+IC) completed in 0.843s
[INFO] [8.53s] HeatTransfer converged: iter=220 res=9.63175e-11
[INFO] [8.54s] Coupling iteration 3, residual = 7.08082e-06
[INFO] [8.97s] Structural assemble completed in 0.435s
[INFO] [17.85s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [17.85s] Linear solve (UMFPACK) completed in 8.877s
[INFO] [17.85s] StructuralSolver: displacement norm = 0.00592757
[INFO] [17.85s] Coupling solve completed in 15.914s
[INFO] [17.85s] Coupling converged in 3 iterations
[INFO] [17.85s] Potential range: [-1.50514e-13, 0.02] V
[INFO] [17.85s] Temperature range: [322.185, 330.041] K
[INFO] [17.85s] Temperature range: [49.0352, 56.8909] C
[INFO] [17.85s] Max displacement magnitude: 5.07861e-05 m
[INFO] [17.86s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [17.96s] Exported VTU results to results/busbar_results.vtu
[INFO] [17.96s] Exported VTU results to: results/busbar_results.vtu
[INFO] [18.03s] Exported results to results/mpfem_result.txt
[INFO] [18.03s] Exported results to: results/mpfem_result.txt
[INFO] [18.03s] === Example completed successfully! ===
```
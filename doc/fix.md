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

* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码。
* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的数性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。
```
 HUAWEI    mpfem  dev ↑2  ~2 |  ~1   1m 10.304s⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                  pwsh   98  18:14:26 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [0ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [175ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [246ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [247ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [248ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [250ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [250ms] Building electrostatics solver, order = 2
[INFO] [474ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [475ms] Building heat transfer solver, order = 2
[INFO] [707ms] HeatTransferSolver: 49889 DOFs
[INFO] [707ms] Building structural solver, order = 2
[INFO] [2.32s] StructuralSolver: 149667 DOFs
[INFO] [2.32s] Joule heating domains: 7 domains
[INFO] [2.32s] Thermal expansion coupling enabled
[INFO] [2.32s] Running coupled electro-thermal solve...
[INFO] [2.41s] Electrostatics assemble completed in 0.094s
[INFO] [3.34s] Linear solve (CG+IC) completed in 0.924s
[INFO] [3.34s] Electrostatics converged: iter=250 res=8.45618e-11
[INFO] [3.52s] HeatTransfer assemble completed in 0.178s
[INFO] [4.46s] Linear solve (CG+IC) completed in 0.941s
[INFO] [4.46s] HeatTransfer converged: iter=264 res=7.79472e-11
[INFO] [4.46s] Coupling iteration 1, residual = 1
[INFO] [4.54s] Electrostatics assemble completed in 0.082s
[INFO] [5.45s] Linear solve (CG+IC) completed in 0.910s
[INFO] [5.45s] Electrostatics converged: iter=234 res=9.46602e-11
[INFO] [5.78s] HeatTransfer assemble completed in 0.327s
[INFO] [6.77s] Linear solve (CG+IC) completed in 0.990s
[INFO] [6.77s] HeatTransfer converged: iter=245 res=9.30345e-11
[INFO] [6.77s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.86s] Electrostatics assemble completed in 0.082s
[INFO] [7.69s] Linear solve (CG+IC) completed in 0.838s
[INFO] [7.69s] Electrostatics converged: iter=209 res=8.0454e-11
[INFO] [7.92s] HeatTransfer assemble completed in 0.221s
[INFO] [8.78s] Linear solve (CG+IC) completed in 0.860s
[INFO] [8.78s] HeatTransfer converged: iter=221 res=9.18655e-11
[INFO] [8.78s] Coupling iteration 3, residual = 7.08082e-06
[INFO] [9.23s] Structural assemble completed in 0.452s
[INFO] [17.41s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [17.42s] Linear solve (UMFPACK) completed in 8.186s
[INFO] [17.42s] StructuralSolver: displacement norm = 0.00592757
[INFO] [17.42s] Coupling solve completed in 15.098s
[INFO] [17.42s] Coupling converged in 3 iterations
[INFO] [17.42s] Potential range: [-1.49787e-13, 0.02] V
[INFO] [17.42s] Temperature range: [322.185, 330.041] K
[INFO] [17.42s] Temperature range: [49.0352, 56.8909] C
[INFO] [17.42s] Max displacement magnitude: 5.07861e-05 m
[INFO] [17.43s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [17.52s] Exported VTU results to results/busbar_results.vtu
[INFO] [17.52s] Exported VTU results to: results/busbar_results.vtu
[INFO] [17.57s] Exported results to results/mpfem_result.txt
[INFO] [17.57s] Exported results to: results/mpfem_result.txt
[INFO] [17.57s] === Example completed successfully! ===
```
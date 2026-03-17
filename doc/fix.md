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
 HUAWEI    mpfem  dev ≡  ~3   284ms⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                                               19:02:11 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [1ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [2ms] Loaded case definition: busbar with 3 physics fields
[INFO] [2ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [3ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [204ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [291ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [292ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [292ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [293ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [294ms] Building electrostatics solver, order = 2
[INFO] [472ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [502ms] Building heat transfer solver, order = 2
[INFO] [678ms] HeatTransferSolver: 49889 DOFs
[INFO] [678ms] Building structural solver, order = 2
[INFO] [2.15s] StructuralSolver: 149667 DOFs
[INFO] [2.15s] Joule heating domains: 7 domains
[INFO] [2.15s] Thermal expansion coupling enabled
[INFO] [2.15s] Running coupled electro-thermal solve...
[INFO] [2.26s] Electrostatics assemble completed in 0.112s
[INFO] [3.24s] Linear solve (CG+IC) completed in 0.977s
[INFO] [3.24s] Electrostatics converged: iter=250 res=8.63277e-11
[INFO] [3.46s] HeatTransfer assemble completed in 0.217s
[INFO] [4.51s] Linear solve (CG+IC) completed in 1.048s
[INFO] [4.51s] HeatTransfer converged: iter=264 res=7.78405e-11
[INFO] [4.51s] Coupling iteration 1, residual = 1
[INFO] [4.60s] Electrostatics assemble completed in 0.090s
[INFO] [5.47s] Linear solve (CG+IC) completed in 0.875s
[INFO] [5.48s] Electrostatics converged: iter=234 res=9.39541e-11
[INFO] [5.69s] HeatTransfer assemble completed in 0.209s
[INFO] [6.66s] Linear solve (CG+IC) completed in 0.970s
[INFO] [6.66s] HeatTransfer converged: iter=245 res=9.60876e-11
[INFO] [6.66s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.74s] Electrostatics assemble completed in 0.085s
[INFO] [7.58s] Linear solve (CG+IC) completed in 0.836s
[INFO] [7.58s] Electrostatics converged: iter=209 res=8.01899e-11
[INFO] [7.79s] HeatTransfer assemble completed in 0.212s
[INFO] [8.68s] Linear solve (CG+IC) completed in 0.882s
[INFO] [8.68s] HeatTransfer converged: iter=221 res=9.20871e-11
[INFO] [8.68s] Coupling iteration 3, residual = 7.08083e-06
[INFO] [9.17s] Structural assemble completed in 0.495s
[INFO] [17.79s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [17.79s] Linear solve (UMFPACK) completed in 8.615s
[INFO] [17.79s] StructuralSolver: displacement norm = 0.00592757
[INFO] [17.79s] Coupling solve completed in 15.639s
[INFO] [17.79s] Coupling converged in 3 iterations
[INFO] [17.79s] Potential range: [-1.51074e-13, 0.02] V
[INFO] [17.79s] Temperature range: [322.185, 330.041] K
[INFO] [17.79s] Temperature range: [49.0352, 56.8909] C
[INFO] [17.79s] Max displacement magnitude: 5.07861e-05 m
[INFO] [17.80s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [17.89s] Exported VTU results to results/busbar_results.vtu
[INFO] [17.89s] Exported VTU results to: results/busbar_results.vtu
[INFO] [17.95s] Exported results to results/mpfem_result.txt
[INFO] [17.95s] Exported results to: results/mpfem_result.txt
[INFO] [17.95s] === Example completed successfully! ===
```
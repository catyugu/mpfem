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

* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的数性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。

```text
HUAWEI@LAPTOP-TLRCI986 CLANG64 /e/code/cpp/mpfem
$ ./build/examples/busbar_example.exe ./cases/busbar_order2/
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: ./cases/busbar_order2/
[INFO] [0ms] Reading case from ./cases/busbar_order2//case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from ./cases/busbar_order2//mesh.mphtxt
[INFO] [2ms] Reading mesh from ./cases/busbar_order2//mesh.mphtxt
[INFO] [179ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [289ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [319ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [337ms] Reading materials from ./cases/busbar_order2//material.xml
[INFO] [357ms] Loaded 2 materials from ./cases/busbar_order2//material.xml
[INFO] [381ms] Building electrostatics solver, order = 2
[INFO] [698ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [698ms] Building heat transfer solver, order = 2
[INFO] [904ms] HeatTransferSolver: 49889 DOFs
[INFO] [905ms] Building structural solver, order = 2
[INFO] [2.90s] StructuralSolver: 149667 DOFs
[INFO] [2.90s] Joule heating domains: 7 domains
[INFO] [2.90s] Thermal expansion coupling enabled
[INFO] [2.90s] Running coupled electro-thermal solve...
[INFO] [3.00s] Electrostatics assemble completed in 0.100s
[INFO] [3.85s] Electrostatics converged: iter=249 res=8.94935e-11
[INFO] [4.05s] HeatTransfer assemble completed in 0.192s
[INFO] [4.99s] HeatTransfer converged: iter=264 res=7.78876e-11
[INFO] [4.99s] Coupling iteration 1, residual = 1
[INFO] [5.09s] Electrostatics assemble completed in 0.091s
[INFO] [5.95s] Electrostatics converged: iter=234 res=9.46121e-11
[INFO] [6.15s] HeatTransfer assemble completed in 0.201s
[INFO] [7.04s] HeatTransfer converged: iter=246 res=8.61764e-11
[INFO] [7.04s] Coupling iteration 2, residual = 0.000817744
[INFO] [7.12s] Electrostatics assemble completed in 0.078s
[INFO] [7.85s] Electrostatics converged: iter=208 res=8.57754e-11
[INFO] [8.04s] HeatTransfer assemble completed in 0.181s
[INFO] [8.87s] HeatTransfer converged: iter=221 res=9.75173e-11
[INFO] [8.87s] Coupling iteration 3, residual = 7.08082e-06
[INFO] [8.95s] Electrostatics assemble completed in 0.081s
[INFO] [9.75s] Electrostatics converged: iter=172 res=9.46169e-11
[INFO] [9.93s] HeatTransfer assemble completed in 0.185s
[INFO] [10.64s] HeatTransfer converged: iter=187 res=9.30412e-11
[INFO] [10.64s] Coupling iteration 4, residual = 6.13271e-08
[INFO] [11.14s] Structural assemble completed in 0.502s
[INFO] [21.47s] Coupling solve completed in 18.567s
[INFO] [21.47s] Coupling converged in 4 iterations
[INFO] [21.47s] Potential range: [-2.27518e-13, 0.02] V
[INFO] [21.47s] Temperature range: [322.185, 330.041] K
[INFO] [21.47s] Temperature range: [49.0351, 56.8909] C
[INFO] [21.47s] Max displacement magnitude: 4.9861e-05 m
[INFO] [21.48s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [21.70s] Exported VTU results to results/busbar_results.vtu
[INFO] [21.70s] Results exported to: results/busbar_results.vtu
[INFO] [21.76s] Exported results to results/mpfem_result.txt
[INFO] [21.76s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [21.76s] === Example completed successfully! ===
```
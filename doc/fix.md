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

* 在vtu文件导出时，把位移场导出为向量场而不是只导出幅值，以便调试。
* 在线性求解器中都统一进行计时（利用logger中提供的计时功能）。

## 工作任务2

* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码。
* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的数性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。

```text
HUAWEI@LAPTOP-TLRCI986 CLANG64 /e/code/cpp/mpfem
$ ./build/examples/busbar_example.exe ./cases/busbar_order2
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [2ms] Case directory: ./cases/busbar_order2
[INFO] [3ms] Reading case from ./cases/busbar_order2/case.xml
[INFO] [4ms] Loaded case definition: busbar with 3 physics fields
[INFO] [4ms] Reading mesh from ./cases/busbar_order2/mesh.mphtxt
[INFO] [5ms] Reading mesh from ./cases/busbar_order2/mesh.mphtxt
[INFO] [154ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [212ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [213ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [213ms] Reading materials from ./cases/busbar_order2/material.xml
[INFO] [213ms] Loaded 2 materials from ./cases/busbar_order2/material.xml
[INFO] [214ms] Building electrostatics solver, order = 2
[INFO] [406ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [406ms] Building heat transfer solver, order = 2
[INFO] [595ms] HeatTransferSolver: 49889 DOFs
[INFO] [596ms] Building structural solver, order = 2
[INFO] [2.10s] StructuralSolver: 149667 DOFs
[INFO] [2.10s] Joule heating domains: 7 domains
[INFO] [2.10s] Thermal expansion coupling enabled
[INFO] [2.10s] Running coupled electro-thermal solve...
[INFO] [2.18s] Electrostatics assemble completed in 0.081s
[INFO] [3.04s] Electrostatics converged: iter=250 res=8.60234e-11
[INFO] [3.23s] HeatTransfer assemble completed in 0.179s
[INFO] [4.09s] HeatTransfer converged: iter=263 res=9.46409e-11
[INFO] [4.09s] Coupling iteration 1, residual = 1
[INFO] [4.17s] Electrostatics assemble completed in 0.078s
[INFO] [5.00s] Electrostatics converged: iter=235 res=9.10922e-11
[INFO] [5.18s] HeatTransfer assemble completed in 0.184s
[INFO] [6.06s] HeatTransfer converged: iter=246 res=9.60148e-11
[INFO] [6.06s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.14s] Electrostatics assemble completed in 0.078s
[INFO] [6.97s] Electrostatics converged: iter=209 res=8.02346e-11
[INFO] [7.15s] HeatTransfer assemble completed in 0.174s
[INFO] [7.90s] HeatTransfer converged: iter=221 res=9.22765e-11
[INFO] [7.90s] Coupling iteration 3, residual = 7.08082e-06
[INFO] [8.45s] Structural assemble completed in 0.552s
[INFO] [19.42s] Coupling solve completed in 17.321s
[INFO] [19.42s] Coupling converged in 3 iterations
[INFO] [19.42s] Potential range: [-1.50867e-13, 0.02] V
[INFO] [19.42s] Temperature range: [322.185, 330.041] K
[INFO] [19.42s] Temperature range: [49.0352, 56.8909] C
[INFO] [19.42s] Max displacement magnitude: 4.98606e-05 m
[INFO] [19.43s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [19.61s] Exported VTU results to results/busbar_results.vtu
[INFO] [19.61s] Results exported to: results/busbar_results.vtu
[INFO] [19.67s] Exported results to results/mpfem_result.txt
[INFO] [19.67s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [19.67s] === Example completed successfully! ===
```
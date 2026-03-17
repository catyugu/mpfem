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

* 目前代码有很多深嵌套，而且对于决定于其他场变量的系数的处理也不够优雅，没有能和一般的系数同等放置在域映射表中，而是被单独处理，这是不好的设计范式。
* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码。
* 我注意到代码中存在大量的手动循环数据填充、复制等，这导致了不可忽略的数性能损失，也反映了架构上的一些潜在问题，请你进一步调查。
* 尤其可以注意到，二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入更高性能、收敛更快的迭代求解器。

```
 HUAWEI    mpfem  dev ↑1  ~2 |  ~5   1.865s⠀   python .\scripts\compare_comsol_results.py ./results/mpfem_result.txt ./cases/busbar/result.txt       5:47 
field   L2      Linf    max_relative    L2_relative
V       1.398699e-08    2.095438e-08    2.811708e-06    1.912434e-06
T       2.553193e-05    2.817878e-05    8.711799e-08    7.883412e-08
disp    2.420364e-11    4.662172e-11    9.686745e-07    8.734414e-07
 HUAWEI    mpfem  dev ↑1  ~2 |  ~5   233ms⠀   cmake --build build -j4                                                                               45:51 
[26/26] Linking CXX executable examples\busbar_example.exe
 HUAWEI    mpfem  dev ↑1  ~5   17.557s⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                                              21:46:37 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [1ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [2ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [166ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [251ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [252ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [253ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [253ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [254ms] Building electrostatics solver, order = 2
[INFO] [427ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [428ms] Building heat transfer solver, order = 2
[INFO] [602ms] HeatTransferSolver: 49889 DOFs
[INFO] [603ms] Building structural solver, order = 2
[INFO] [1.94s] StructuralSolver: 149667 DOFs
[INFO] [1.95s] Joule heating domains: 7 domains
[INFO] [1.95s] Thermal expansion coupling enabled
[INFO] [1.95s] Running coupled electro-thermal solve...
[INFO] [2.04s] Electrostatics assemble completed in 0.091s
[INFO] [3.00s] Linear solve (CG+IC) completed in 0.962s
[INFO] [3.00s] Electrostatics converged: iter=249 res=9.92058e-11
[INFO] [3.21s] HeatTransfer assemble completed in 0.206s
[INFO] [4.16s] Linear solve (CG+IC) completed in 0.955s
[INFO] [4.16s] HeatTransfer converged: iter=264 res=8.69092e-11
[INFO] [4.17s] Coupling iteration 1, residual = 1
[INFO] [4.25s] Electrostatics assemble completed in 0.082s
[INFO] [5.08s] Linear solve (CG+IC) completed in 0.826s
[INFO] [5.08s] Electrostatics converged: iter=234 res=9.42881e-11
[INFO] [5.27s] HeatTransfer assemble completed in 0.190s
[INFO] [6.18s] Linear solve (CG+IC) completed in 0.912s
[INFO] [6.18s] HeatTransfer converged: iter=247 res=7.94275e-11
[INFO] [6.18s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.27s] Electrostatics assemble completed in 0.083s
[INFO] [7.07s] Linear solve (CG+IC) completed in 0.799s
[INFO] [7.07s] Electrostatics converged: iter=209 res=8.07957e-11
[INFO] [7.26s] HeatTransfer assemble completed in 0.199s
[INFO] [8.13s] Linear solve (CG+IC) completed in 0.868s
[INFO] [8.13s] HeatTransfer converged: iter=221 res=9.60913e-11
[INFO] [8.13s] Coupling iteration 3, residual = 7.08083e-06
[INFO] [8.64s] Structural assemble completed in 0.507s
[INFO] [18.17s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [18.17s] Linear solve (UMFPACK) completed in 9.524s
[INFO] [18.17s] StructuralSolver: displacement norm = 0.00592757
[INFO] [18.17s] Coupling solve completed in 16.221s
[INFO] [18.17s] Coupling converged in 3 iterations
[INFO] [18.17s] Potential range: [-1.48101e-13, 0.02] V
[INFO] [18.17s] Temperature range: [322.185, 330.041] K
[INFO] [18.17s] Temperature range: [49.0352, 56.8909] C
[INFO] [18.17s] Max displacement magnitude: 5.07861e-05 m
[INFO] [18.18s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [18.27s] Exported VTU results to results/busbar_results.vtu
[INFO] [18.27s] Exported VTU results to: results/busbar_results.vtu
[INFO] [18.33s] Exported results to results/mpfem_result.txt
[INFO] [18.33s] Exported results to: results/mpfem_result.txt
[INFO] [18.33s] === Example completed successfully! ===
```
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

* 二阶问题求解的性能较低，需要：1、优化问题构建的时间（现在需要约两秒）；2、引入比eigen更高性能的迭代求解器。

```text
$ ./build/examples/busbar_example.exe ./cases/busbar_order2/
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [1ms] Case directory: ./cases/busbar_order2/
[INFO] [2ms] Reading case from ./cases/busbar_order2//case.xml
[INFO] [3ms] Loaded case definition: busbar with 3 physics fields
[INFO] [3ms] Reading mesh from ./cases/busbar_order2//mesh.mphtxt
[INFO] [4ms] Reading mesh from ./cases/busbar_order2//mesh.mphtxt
[INFO] [163ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [231ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [232ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [233ms] Reading materials from ./cases/busbar_order2//material.xml
[INFO] [234ms] Loaded 2 materials from ./cases/busbar_order2//material.xml
[INFO] [234ms] Building electrostatics solver, order = 2
[INFO] [444ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [445ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [446ms] Domain 2 (mat2): sigma = 740700
[INFO] [447ms] Domain 3 (mat2): sigma = 740700
[INFO] [448ms] Domain 4 (mat2): sigma = 740700
[INFO] [449ms] Domain 5 (mat2): sigma = 740700
[INFO] [449ms] Domain 6 (mat2): sigma = 740700
[INFO] [449ms] Domain 7 (mat2): sigma = 740700
[INFO] [450ms] Building heat transfer solver, order = 2
[INFO] [658ms] HeatTransferSolver: 49889 DOFs
[INFO] [658ms] Building structural solver, order = 2
[INFO] [2.09s] StructuralSolver: 149667 DOFs
[INFO] [2.09s] Domain 1 (mat1): E = 110, nu = 0.35
[INFO] [2.09s] Domain 2 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Domain 3 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Domain 4 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Domain 5 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Domain 6 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Domain 7 (mat2): E = 105, nu = 0.33
[INFO] [2.09s] Joule heating domains: 7 domains
[INFO] [2.09s] Thermal expansion coupling enabled
[INFO] [2.09s] Domain 1 (mat1): temp-dep sigma, rho0 = 1.72e-08, alpha = 0.0039
[INFO] [2.09s] Running coupled electro-thermal solve...
[INFO] [2.17s] Electrostatics assemble completed in 0.081s
[INFO] [3.02s] Electrostatics converged: iter=250 res=8.77735e-11
[INFO] [3.20s] HeatTransfer assemble completed in 0.172s
[INFO] [4.13s] HeatTransfer converged: iter=264 res=7.78181e-11
[INFO] [4.13s] Coupling iteration 1, residual = 1
[INFO] [4.21s] Electrostatics assemble completed in 0.080s
[INFO] [5.07s] Electrostatics converged: iter=235 res=9.8381e-11
[INFO] [5.23s] HeatTransfer assemble completed in 0.160s
[INFO] [6.08s] HeatTransfer converged: iter=245 res=9.40435e-11
[INFO] [6.08s] Coupling iteration 2, residual = 0.000817744
[INFO] [6.51s] Structural assemble completed in 0.435s
[INFO] [15.88s] Coupling solve completed in 13.785s
[INFO] [15.88s] Coupling converged in 2 iterations
[INFO] [15.88s] Potential range: [-1.45745e-13, 0.02] V
[INFO] [15.88s] Temperature range: [322.183, 330.037] K
[INFO] [15.88s] Temperature range: [49.033, 56.8875] C
[INFO] [15.88s] Max displacement magnitude: 0.00145326 m
[INFO] [15.88s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [16.07s] Exported VTU results to results/busbar_results.vtu
[INFO] [16.07s] Results exported to: results/busbar_results.vtu
[INFO] [16.12s] Exported results to results/mpfem_result.txt
[INFO] [16.12s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [16.12s] === Example completed successfully! ===
```
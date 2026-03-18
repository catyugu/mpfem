# 大纲

依据@prompt.md ，适当参考学习external/mfem中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 完成一块工作任务后：
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。
  * 提交一次代码，然后继续下一块工作。

## 工作任务1

* cmake构建系统的模块化程度不足，很多依赖套叠嵌合在一起，寻找外部依赖的方式又各不相同，看起来非常混乱。这是首要的需要改善的部分。
* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码（例如span，range等等）。
* 当前的Coefficient机制比较僵硬，标量、向量、矩阵形状的Coefficient需要分别定义，这是不好的设计，或许你应该结合C++20的concept等特性，做一个类型安全的张量类统一设计？此外，有些材料属性在配置文件中按照矩阵定义，则也应该读成矩阵而不是单个数字（这样对于以后可能遇到的各向异性材料更通用）。
* 日志默认级别应该由编译选项指定，Debug模式下默认为Info，Release模式下默认为Warning，但也允许用户代码中运行时手动修改。

```
 HUAWEI    mpfem  dev ≡  ~1 |  ~6  1   16.957s⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2        pwsh   98  01:07:37 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [1ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [2ms] Loaded case definition: busbar with 3 physics fields
[INFO] [2ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [3ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [198ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [285ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [287ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [288ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [288ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [289ms] Building electrostatics solver, order = 2
[INFO] [293ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [294ms] Building heat transfer solver, order = 2
[INFO] [300ms] HeatTransferSolver: 49889 DOFs
[INFO] [300ms] Building structural solver, order = 2
[INFO] [308ms] StructuralSolver: 149667 DOFs
[INFO] [310ms] Joule heating domains: 7 domains
[INFO] [310ms] Thermal expansion coupling enabled
[INFO] [312ms] Running coupled electro-thermal solve...
[INFO] [406ms] Electrostatics assemble completed in 0.094s
[INFO] [1.86s] [UMFPACK] Solve successful, solution norm: 1.61365
[INFO] [1.86s] Linear solve (UMFPACK) completed in 1.457s
[INFO] [1.86s] Electrostatics solver converged!
[INFO] [2.06s] HeatTransfer assemble completed in 0.193s
[INFO] [3.21s] [UMFPACK] Solve successful, solution norm: 72119.9
[INFO] [3.21s] Linear solve (UMFPACK) completed in 1.151s
[INFO] [3.21s] HeatTransfer converged!
[INFO] [3.21s] Coupling iteration 1, residual = 1
[INFO] [3.31s] Electrostatics assemble completed in 0.095s
[INFO] [3.42s] [UMFPACK] Solve successful, solution norm: 1.62178
[INFO] [3.42s] Linear solve (UMFPACK) completed in 0.113s
[INFO] [3.42s] Electrostatics solver converged!
[INFO] [3.71s] HeatTransfer assemble completed in 0.290s
[INFO] [3.81s] [UMFPACK] Solve successful, solution norm: 72061.2
[INFO] [3.81s] Linear solve (UMFPACK) completed in 0.098s
[INFO] [3.81s] HeatTransfer converged!
[INFO] [3.81s] Coupling iteration 2, residual = 0.00081761
[INFO] [3.93s] Electrostatics assemble completed in 0.113s
[INFO] [4.00s] [UMFPACK] Solve successful, solution norm: 1.62171
[INFO] [4.00s] Linear solve (UMFPACK) completed in 0.076s
[INFO] [4.00s] Electrostatics solver converged!
[INFO] [4.21s] HeatTransfer assemble completed in 0.202s
[INFO] [4.29s] [UMFPACK] Solve successful, solution norm: 72061.7
[INFO] [4.30s] Linear solve (UMFPACK) completed in 0.089s
[INFO] [4.30s] HeatTransfer converged!
[INFO] [4.30s] Coupling iteration 3, residual = 7.07832e-06
[INFO] [4.93s] Structural assemble completed in 0.631s
[INFO] [13.90s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [13.90s] Linear solve (UMFPACK) completed in 8.969s
[INFO] [13.90s] StructuralSolver: displacement norm = 0.00592757
[INFO] [13.90s] Coupling solve completed in 13.589s
[INFO] [13.90s] Coupling converged in 3 iterations
[INFO] [13.90s] Potential range: [0, 0.02] V
[INFO] [13.90s] Temperature range: [322.185, 330.042] K
[INFO] [13.90s] Temperature range: [49.0351, 56.8922] C
[INFO] [13.90s] Max displacement magnitude: 5.0786e-05 m
[INFO] [13.91s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [14.00s] Exported VTU results to results/busbar_results.vtu
[INFO] [14.00s] Exported VTU results to: results/busbar_results.vtu
[INFO] [14.05s] Exported results to results/mpfem_result.txt
[INFO] [14.05s] Exported results to: results/mpfem_result.txt
[INFO] [14.05s] === Example completed successfully! ===
```
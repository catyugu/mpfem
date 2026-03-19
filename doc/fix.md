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

* 目前的求解性能太差劲了，可能需要采用更好预条件下的迭代求解器。
  
```
 HUAWEI    mpfem  dev ↑1  2   20.337s⠀   .\build-llvm\examples\busbar_example.exe .\cases\busbar_order2\
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [1ms] Case directory: .\cases\busbar_order2\
[INFO] [1ms] Reading case from .\cases\busbar_order2\/case.xml
[INFO] [2ms] Loaded case definition: busbar with 3 physics fields
[INFO] [3ms] Reading mesh from .\cases\busbar_order2\/mesh.mphtxt
[INFO] [3ms] Reading mesh from .\cases\busbar_order2\/mesh.mphtxt
[INFO] [180ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [255ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [257ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [257ms] Reading materials from .\cases\busbar_order2\/material.xml
[INFO] [258ms] Loaded 2 materials from .\cases\busbar_order2\/material.xml
[INFO] [258ms] Building electrostatics solver, order = 2
[INFO] [262ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [262ms] Building heat transfer solver, order = 2
[INFO] [266ms] HeatTransferSolver: 49889 DOFs
[INFO] [266ms] Building structural solver, order = 2
[INFO] [273ms] StructuralSolver: 149667 DOFs
[INFO] [273ms] Joule heating domains: 7 domains
[INFO] [274ms] Thermal expansion coupling enabled
[INFO] [274ms] Running coupled electro-thermal solve...
[INFO] [355ms] Electrostatics assemble completed in 0.081s
[INFO] [2.02s] [UMFPACK] Solve successful, solution norm: 1.61365
[INFO] [2.02s] Linear solve (UMFPACK) completed in 1.669s
[INFO] [2.03s] Electrostatics solver converged!
[INFO] [2.23s] HeatTransfer assemble completed in 0.200s
[INFO] [3.83s] [UMFPACK] Solve successful, solution norm: 72119.9
[INFO] [3.84s] Linear solve (UMFPACK) completed in 1.608s
[INFO] [3.84s] HeatTransfer converged!
[INFO] [3.84s] Coupling iteration 1, residual = 1
[INFO] [3.92s] Electrostatics assemble completed in 0.080s
[INFO] [5.49s] [UMFPACK] Solve successful, solution norm: 1.62185
[INFO] [5.49s] Linear solve (UMFPACK) completed in 1.568s
[INFO] [5.49s] Electrostatics solver converged!
[INFO] [5.65s] HeatTransfer assemble completed in 0.162s
[INFO] [7.28s] [UMFPACK] Solve successful, solution norm: 72061.2
[INFO] [7.28s] Linear solve (UMFPACK) completed in 1.624s
[INFO] [7.28s] HeatTransfer converged!
[INFO] [7.28s] Coupling iteration 2, residual = 0.000817744
[INFO] [7.36s] Electrostatics assemble completed in 0.083s
[INFO] [8.77s] [UMFPACK] Solve successful, solution norm: 1.62178
[INFO] [8.77s] Linear solve (UMFPACK) completed in 1.407s
[INFO] [8.77s] Electrostatics solver converged!
[INFO] [8.91s] HeatTransfer assemble completed in 0.145s
[INFO] [10.24s] [UMFPACK] Solve successful, solution norm: 72061.7
[INFO] [10.24s] Linear solve (UMFPACK) completed in 1.328s
[INFO] [10.24s] HeatTransfer converged!
[INFO] [10.24s] Coupling iteration 3, residual = 7.08083e-06
[INFO] [10.72s] Structural assemble completed in 0.475s
[INFO] [20.63s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [20.63s] Linear solve (UMFPACK) completed in 9.908s
[INFO] [20.63s] StructuralSolver: displacement norm = 0.00592757
[INFO] [20.63s] Coupling solve completed in 20.354s
[INFO] [20.63s] Coupling converged in 3 iterations
[INFO] [20.63s] Potential range: [0, 0.02] V
[INFO] [20.63s] Temperature range: [322.185, 330.041] K
[INFO] [20.63s] Temperature range: [49.0352, 56.8909] C
[INFO] [20.63s] Max displacement magnitude: 5.07861e-05 m
[INFO] [20.64s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [20.74s] Exported VTU results to results/busbar_results.vtu
[INFO] [20.74s] Exported VTU results to: results/busbar_results.vtu
[INFO] [20.80s] Exported results to results/mpfem_result.txt
[INFO] [20.80s] Exported results to: results/mpfem_result.txt
[INFO] [20.80s] === Example completed successfully! ===
```

## 工作任务2

* 可以使用泛型来简化Coefficient相关的设计（当然在使用处需配合concept来保证安全），以免于区分标量、向量、矩阵Coefficient的需要。
* 这样，Coefficient在存储、设置和传递时都必须擦除其派生类型。
* 请完全用新的设计取代老的设计！在外面的应用层，如输入材料时候，如果材料属性是矩阵就使用矩阵，是标量就用标量，而不需要任何硬编码的操作。但是在积分器里可以进行concept限制（例如DiffusionIntegrator只能使用标量或者矩阵型的Coefficient等等）
* 不要进行任何向后兼容的别名。
* 设计架构的时候请考虑未来扩展到瞬态求解时候的需要。
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

* 可以使用泛型来简化Coefficient相关的设计（当然在使用处需配合concept来保证安全），以免于区分标量、向量、矩阵Coefficient的需要。
* 这样，Coefficient在存储、设置和传递时都必须擦除其派生类型，只需要在创建时指定，使用时使用concept限制（例如DiffusionIntegrator只能使用标量或者矩阵型的Coefficient等等）。只需要一个统一的泛型基类就够了，不管是标量，向量还是矩阵，统统可以处理！
* 请完全用新的设计取代老的设计！在外面的应用层不再区分Coefficient是Scalar还是Vector或是Matrix。
* 设计架构的时候请考虑未来扩展到瞬态求解时候的需要。
* 重新考虑架构，似乎CouplingManager是多余的，我们引入Problem的概念，PhysicsProblemBuilder负责构造Problem（稳态/瞬态/特征值），Problem层持有所有原始数据（场值，系数等），下游则是单场的离散和组装逻辑。

```
 HUAWEI    mpfem  main ≡  2   85ms⠀   .\build-llvm\examples\busbar_example.exe .\cases\busbar_order2\                                       0:59 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2\
[INFO] [0ms] Reading case from .\cases\busbar_order2\/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2\/mesh.mphtxt
[INFO] [1ms] Reading mesh from .\cases\busbar_order2\/mesh.mphtxt
[INFO] [154ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [218ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [219ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [219ms] Reading materials from .\cases\busbar_order2\/material.xml
[INFO] [220ms] Loaded 2 materials from .\cases\busbar_order2\/material.xml
[INFO] [220ms] Building electrostatics solver, order = 2
[INFO] [224ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [225ms] Building heat transfer solver, order = 2
[INFO] [229ms] HeatTransferSolver: 49889 DOFs
[INFO] [229ms] Building structural solver, order = 2
[INFO] [235ms] StructuralSolver: 149667 DOFs
[INFO] [235ms] Joule heating domains: 7 domains
[INFO] [236ms] Thermal expansion coupling enabled
[INFO] [236ms] Running coupled electro-thermal solve...
[INFO] [310ms] Electrostatics assemble completed in 0.073s
[INFO] [1.11s] [UMFPACK] Solve successful, solution norm: 1.61365
[INFO] [1.11s] Linear solve (UMFPACK) completed in 0.803s
[INFO] [1.11s] Electrostatics solver converged!
[INFO] [1.25s] HeatTransfer assemble completed in 0.138s
[INFO] [2.06s] [UMFPACK] Solve successful, solution norm: 72119.9
[INFO] [2.06s] Linear solve (UMFPACK) completed in 0.813s
[INFO] [2.06s] HeatTransfer converged!
[INFO] [2.07s] Coupling iteration 1, residual = 1
[INFO] [2.13s] Electrostatics assemble completed in 0.059s
[INFO] [3.01s] [UMFPACK] Solve successful, solution norm: 1.62185
[INFO] [3.01s] Linear solve (UMFPACK) completed in 0.888s
[INFO] [3.01s] Electrostatics solver converged!
[INFO] [3.15s] HeatTransfer assemble completed in 0.139s
[INFO] [3.97s] [UMFPACK] Solve successful, solution norm: 72061.2
[INFO] [3.97s] Linear solve (UMFPACK) completed in 0.814s
[INFO] [3.97s] HeatTransfer converged!
[INFO] [3.97s] Coupling iteration 2, residual = 0.000817744
[INFO] [4.04s] Electrostatics assemble completed in 0.069s
[INFO] [4.97s] [UMFPACK] Solve successful, solution norm: 1.62178
[INFO] [4.97s] Linear solve (UMFPACK) completed in 0.931s
[INFO] [4.97s] Electrostatics solver converged!
[INFO] [5.12s] HeatTransfer assemble completed in 0.149s
[INFO] [5.93s] [UMFPACK] Solve successful, solution norm: 72061.7
[INFO] [5.93s] Linear solve (UMFPACK) completed in 0.815s
[INFO] [5.93s] HeatTransfer converged!
[INFO] [5.93s] Coupling iteration 3, residual = 7.08083e-06
[INFO] [6.43s] Structural assemble completed in 0.493s
[INFO] [12.91s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [12.91s] Linear solve (UMFPACK) completed in 6.477s
[INFO] [12.91s] StructuralSolver: displacement norm = 0.00592757
[INFO] [12.91s] Coupling solve completed in 12.671s
[INFO] [12.91s] Coupling converged in 3 iterations
[INFO] [12.91s] Potential range: [0, 0.02] V
[INFO] [12.91s] Temperature range: [322.185, 330.041] K
[INFO] [12.91s] Temperature range: [49.0352, 56.8909] C
[INFO] [12.91s] Max displacement magnitude: 5.07861e-05 m
[INFO] [12.93s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [13.03s] Exported VTU results to results/busbar_results.vtu
[INFO] [13.03s] Exported VTU results to: results/busbar_results.vtu
[INFO] [13.09s] Exported results to results/mpfem_result.txt
[INFO] [13.09s] Exported results to: results/mpfem_result.txt
[INFO] [13.09s] === Example completed successfully! ===
```
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

* 当前求解结果有奇怪的误差，一阶求解结果非常准确，二阶求解热场结果非常准确，但是电场和位移场却有固有偏差，而且不能通过调小非线性迭代的容差来消除。请调查原因,并修复解决精度问题。
* 这个问题在数次提交之前尚不存在，我也不知道是哪一次重构导致了这个问题，你可以翻看git历史查看以前的代码，观察导致问题的原因。
* 网格节点排序没有问题，不要怀疑这个。

```text
python .\scripts\compare_comsol_results.py ./results/mpfem_result.txt ./cases/busbar_order2/result.txt
field   L2      Linf    max_relative    L2_relative
V       5.677082e-07    1.052203e-06    1.333226e-04    7.778853e-05
T       1.837483e-04    1.264829e-03    3.833270e-06    5.695235e-07
disp    5.601185e-09    1.851248e-08    5.414331e-03    2.110116e-04
```

## 工作任务2

* 可以引入 C++ 20 的新特性用于有效简化或者抽象代码（例如span，range，concept等等）。
* 可以使用泛型来简化Coefficient相关的设计（当然在使用处需配合concept来保证安全），以免于区分标量、向量、矩阵Coefficient的需要。
* 这样，Coefficient在存储、设置和传递时都可以擦除其派生类型，只需要在创建时指定，使用时使用concept限制（例如DiffusionIntegrator只能使用标量或者矩阵型的Coefficient等等）。只需要一个统一的泛型基类就够了，不管是标量，向量还是矩阵，统统可以处理！
* 请完全用新的设计取代老的设计！在外面的应用层不再区分Coefficient是Scalar还是Vector或是Matrix。
* 此外，structrural_solver直接持有积分器，这是不好的设计，物理场不应该直接持有积分器，而是应该持有项/系数。
* 设计架构的时候请考虑未来扩展到瞬态求解时候的需要。
* 重新考虑架构，似乎CouplingManager是多余的，我们引入Problem的概念，PhysicsProblemBuilder负责构造Problem（稳态/瞬态/特征值），Problem层持有所有原始数据（场值，系数等），下游则是单场的离散和组装逻辑。

```
 HUAWEI    mpfem  dev ≡  1   137ms⠀   ./build/examples/busbar_example.exe .\cases\busbar_order2                                             56:37 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: .\cases\busbar_order2
[INFO] [0ms] Reading case from .\cases\busbar_order2/case.xml
[INFO] [0ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [1ms] Reading mesh from .\cases\busbar_order2/mesh.mphtxt
[INFO] [149ms] Mesh loaded: 49889 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [218ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [220ms] Mesh loaded: 49889 vertices, 31021 elements
[INFO] [220ms] Reading materials from .\cases\busbar_order2/material.xml
[INFO] [220ms] Loaded 2 materials from .\cases\busbar_order2/material.xml
[INFO] [220ms] Building electrostatics solver, order = 2
[INFO] [224ms] ElectrostaticsSolver: 49889 DOFs
[INFO] [224ms] Building heat transfer solver, order = 2
[INFO] [228ms] HeatTransferSolver: 49889 DOFs
[INFO] [228ms] Building structural solver, order = 2
[INFO] [235ms] StructuralSolver: 149667 DOFs
[INFO] [235ms] Joule heating domains: 7 domains
[INFO] [235ms] Thermal expansion coupling enabled
[INFO] [236ms] Running coupled electro-thermal solve...
[INFO] [315ms] Electrostatics assemble completed in 0.079s
[INFO] [1.25s] [UMFPACK] Solve successful, solution norm: 1.61365
[INFO] [1.25s] Linear solve (UMFPACK) completed in 0.931s
[INFO] [1.25s] Electrostatics solver converged!
[INFO] [1.41s] HeatTransfer assemble completed in 0.160s
[INFO] [2.31s] [UMFPACK] Solve successful, solution norm: 72119.9
[INFO] [2.31s] Linear solve (UMFPACK) completed in 0.908s
[INFO] [2.31s] HeatTransfer converged!
[INFO] [2.32s] Coupling iteration 1, residual = 1
[INFO] [2.40s] Electrostatics assemble completed in 0.082s
[INFO] [2.46s] [UMFPACK] Solve successful, solution norm: 1.62178
[INFO] [2.46s] Linear solve (UMFPACK) completed in 0.066s
[INFO] [2.46s] Electrostatics solver converged!
[INFO] [2.62s] HeatTransfer assemble completed in 0.155s
[INFO] [2.69s] [UMFPACK] Solve successful, solution norm: 72061.2
[INFO] [2.69s] Linear solve (UMFPACK) completed in 0.073s
[INFO] [2.69s] HeatTransfer converged!
[INFO] [2.69s] Coupling iteration 2, residual = 0.00081761
[INFO] [2.77s] Electrostatics assemble completed in 0.077s
[INFO] [2.83s] [UMFPACK] Solve successful, solution norm: 1.62171
[INFO] [2.83s] Linear solve (UMFPACK) completed in 0.060s
[INFO] [2.83s] Electrostatics solver converged!
[INFO] [3.00s] HeatTransfer assemble completed in 0.172s
[INFO] [3.06s] [UMFPACK] Solve successful, solution norm: 72061.7
[INFO] [3.06s] Linear solve (UMFPACK) completed in 0.059s
[INFO] [3.06s] HeatTransfer converged!
[INFO] [3.06s] Coupling iteration 3, residual = 7.07832e-06
[INFO] [3.47s] Structural assemble completed in 0.407s
[INFO] [8.68s] [UMFPACK] Solve successful, solution norm: 0.00592757
[INFO] [8.68s] Linear solve (UMFPACK) completed in 5.205s
[INFO] [8.68s] StructuralSolver: displacement norm = 0.00592757
[INFO] [8.68s] Coupling solve completed in 8.442s
[INFO] [8.68s] Coupling converged in 3 iterations
[INFO] [8.68s] Potential range: [0, 0.02] V
[INFO] [8.68s] Temperature range: [322.185, 330.042] K
[INFO] [8.68s] Temperature range: [49.0351, 56.8922] C
[INFO] [8.68s] Max displacement magnitude: 5.0786e-05 m
[INFO] [8.68s] High-order mesh detected: 49889 vertices, 7340 corner vertices
[INFO] [8.77s] Exported VTU results to results/busbar_results.vtu
[INFO] [8.77s] Exported VTU results to: results/busbar_results.vtu
[INFO] [8.87s] Exported results to results/mpfem_result.txt
[INFO] [8.87s] Exported results to: results/mpfem_result.txt
[INFO] [8.87s] === Example completed successfully! ===
```
# 大纲

依据@prompt.md ，适当参考学习external/mfem和external/hpc-fem-playground中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast, friend, mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 解决架构问题后，提交一次代码，然后继续调试精度问题。

## 架构修复

* 单场求解器只应该包含自己的边界条件设置接口、额外积分器设置接口等。不应该直接被注入耦合信息。此外物理场的接口名称应该有物理意义，例如静电场的addDirichletBC应该命名为addVoltageBC，热场的addDirichletBC应该命名为addDirichletBC应该命名为addTemperatureBC，位移场的addDirichletBC应该命名为addDisplacementBC等等。
* 此外，很多场景下，如我们案例中的材料Coefficient值、边界值等是和域/边界编号有关的，并不是一个物理场只有一个，我认为你应该修改所有物理场的接口，每一个材料值应该应该对应一组域/边界选择，而不是一个求解器只持有一个或者两个特定的Coefficient；此外，应该强调，物理场持有的应该是Coefficient的基类，而非什么派生类。你可以改造PWConstCoefficient为PWCoefficient并支持根据域选择不同系数。
* 例如，setConductivity接口应该指定域选择，conductivity接口应该指定域。
* 考虑以后扩展到求解瞬态问题，以及更多参数耦合问题的需求，对代码做必要的重构和抽象。
* 现在各种原始指针，智能指针的混用导致所有权好混乱，请你统一使用智能指针管理变量所有权。
* 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
* 结束后，验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。

## 当前运行结果

```text
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [1ms] Case directory: cases/busbar
[INFO] [2ms] Reading case from cases/busbar/case.xml
[INFO] [3ms] Loaded case definition: busbar with 3 physics fields
[INFO] [3ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [4ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [99ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [174ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [176ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [176ms] Reading materials from cases/busbar/material.xml
[INFO] [177ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [178ms] Building electrostatics solver, order = 1
[INFO] [178ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [178ms] Domain 2 (mat2): sigma = 740700
[INFO] [178ms] Domain 3 (mat2): sigma = 740700
[INFO] [179ms] Domain 4 (mat2): sigma = 740700
[INFO] [179ms] Domain 5 (mat2): sigma = 740700
[INFO] [179ms] Domain 6 (mat2): sigma = 740700
[INFO] [179ms] Domain 7 (mat2): sigma = 740700
[INFO] [196ms] ElectrostaticsSolver: 7340 DOFs
[INFO] [197ms] Building heat transfer solver, order = 1
[INFO] [216ms] HeatTransferSolver: 7340 DOFs
[INFO] [216ms] Building structural solver, order = 1
[INFO] [217ms] Domain 1 (mat1): E = 110, nu = 0.35, alpha_T = 1.7e-05
[INFO] [217ms] Domain 2 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [218ms] Domain 3 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [218ms] Domain 4 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [219ms] Domain 5 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [219ms] Domain 6 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [220ms] Domain 7 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [356ms] StructuralSolver: 22020 DOFs
[INFO] [357ms] Joule heating domains: 7 domains
[INFO] [357ms] Thermal expansion coupling enabled
[INFO] [358ms] Domain 1 (mat1): temp-dep sigma, rho0 = 1.72e-08, alpha = 0.0039
[INFO] [358ms] Running coupled electro-thermal solve...
[INFO] [461ms] [EigenSparseLU] Solve successful, solution norm: 0.623237
[INFO] [462ms] Electrostatics converged: iter=1 res=3.7429e-13
[INFO] [659ms] [EigenSparseLU] Solve successful, solution norm: 27772.3
[INFO] [660ms] HeatTransfer converged: iter=1 res=4.80868e-11
[INFO] [660ms] Coupling iteration 1, residual = 1
[INFO] [757ms] [EigenSparseLU] Solve successful, solution norm: 0.626625
[INFO] [758ms] Electrostatics converged: iter=1 res=3.52088e-13
[INFO] [892ms] [EigenSparseLU] Solve successful, solution norm: 27746.8
[INFO] [893ms] HeatTransfer converged: iter=1 res=4.80152e-11
[INFO] [893ms] Coupling iteration 2, residual = 0.000920725
[INFO] [1.78s] [EigenSparseLU] Solve successful, solution norm: 0.00237386
[INFO] [1.78s] StructuralSolver: displacement norm = 0.00237386
[INFO] [1.78s] Coupling converged in 2 iterations
[INFO] [1.78s] Potential range: [0, 0.02] V
[INFO] [1.78s] Temperature range: [323.421, 331.43] K
[INFO] [1.78s] Temperature range: [50.2706, 58.2795] C
[INFO] [1.78s] Max displacement magnitude: 5.28686e-05 m
[INFO] [1.83s] Exported VTU results to results/busbar_results.vtu
[INFO] [1.83s] Results exported to: results/busbar_results.vtu
[INFO] [1.88s] Exported results to results/mpfem_result.txt
[INFO] [1.88s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [1.88s] === Example completed successfully! ===
```

* 求解正确性也可疑，你需要将结果dump成输出文件。在相同的设置下我们的精度应该达到这个级别：

```text
python3 scripts/compare_comsol_results.py cases/busbar/result.txt res
ults/busbar/mpfem_result.txt
field   L2      Linf    max_relative    L2_relative
V       2.148742e-09    5.572466e-08    5.224217e-06    2.965192e-07
T       2.364130e-06    7.245394e-05    2.217743e-07    7.310244e-09
disp    9.269475e-09    3.351310e-08    9.775915e-03    3.403081e-04
```

当前的误差是：

```text
python .\scripts\compare_comsol_results.py ./results/mpfem_result.txt ./cases/busbar/result.txt
field   L2      Linf    max_relative    L2_relative
V       6.631425e-07    1.193437e-06    1.502601e-04    9.066653e-05
T       2.772982e-03    4.183303e-03    1.262226e-05    8.562125e-06
disp    2.466711e-09    4.695249e-09    1.109621e-04    8.902466e-05
```

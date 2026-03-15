# 大纲

依据@prompt.md ，适当参考学习external/mfem和external/hpc-fem-playground中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 我的建议是，在底层完全复现且只复现hpfem中用到的mfem的功能，不要创造太多冗余的接口或者抽象。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast, friend, mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 解决架构问题后，提交一次代码，然后继续调试精度问题。

## 架构修复

* 单场求解器只应该包含自己的边界条件设置接口、源项设置接口等，不应该包含耦合场相关信息等。
* 此外，很多场景下，如我们案例中的场景Coefficient值、边界值等是和域/边界编号有关的，并不是一个物理场只有一个，我认为你应该修改所有物理场的接口，每一个材料值应该应该对应一组域/边界选择，而不是一个求解器只持有一个或者两个特定的Coefficient；此外，应该强调，物理场持有的应该是Coefficient的基类，而非什么派生类。
* 接口示例：`setElectricConductivity(const std::vector<int>& selection, const Coefficient *coeff);`
* 关于基函数的vdim的处理应该中心化，在一个合理的地方统一管理基函数的值维数问题。
* 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
* 结束后，验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。

## 当前运行结果

```text
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [1ms] Case directory: cases/busbar
[INFO] [2ms] Reading case from cases/busbar/case.xml
[INFO] [3ms] Loaded case definition: busbar with 3 physics fields
[INFO] [5ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [6ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [106ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [167ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [168ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [169ms] Reading materials from cases/busbar/material.xml
[INFO] [169ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [169ms] Building electrostatics solver, order = 1
[INFO] [170ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [170ms] Domain 2 (mat2): sigma = 740700
[INFO] [170ms] Domain 3 (mat2): sigma = 740700
[INFO] [170ms] Domain 4 (mat2): sigma = 740700
[INFO] [171ms] Domain 5 (mat2): sigma = 740700
[INFO] [171ms] Domain 6 (mat2): sigma = 740700
[INFO] [171ms] Domain 7 (mat2): sigma = 740700
[INFO] [187ms] ElectrostaticsSolver: 7340 DOFs
[INFO] [187ms] Building heat transfer solver, order = 1
[INFO] [202ms] HeatTransferSolver: 7340 DOFs
[INFO] [202ms] Building structural solver, order = 1
[INFO] [202ms] Domain 1 (mat1): E = 110, nu = 0.35, alpha_T = 1.7e-05
[INFO] [203ms] Domain 2 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [203ms] Domain 3 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [203ms] Domain 4 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [204ms] Domain 5 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [204ms] Domain 6 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [204ms] Domain 7 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [319ms] StructuralSolver: 22020 DOFs
[INFO] [320ms] Joule heating domains: 7 domains
[INFO] [320ms] Thermal expansion coupling enabled
[INFO] [320ms] Domain 1 (mat1): temp-dep sigma, rho0 = 1.72e-08, alpha = 0.0039
[INFO] [321ms] Running coupled electro-thermal solve...
[INFO] [415ms] [EigenSparseLU] Solve successful, solution norm: 0.623237
[INFO] [415ms] Electrostatics converged: iter=1 res=3.77963e-13
[INFO] [649ms] [EigenSparseLU] Solve successful, solution norm: 27772.3
[INFO] [649ms] HeatTransfer converged: iter=1 res=4.81532e-11
[INFO] [649ms] Coupling iteration 1, residual = 1
[INFO] [742ms] [EigenSparseLU] Solve successful, solution norm: 0.626625
[INFO] [742ms] Electrostatics converged: iter=1 res=3.54624e-13
[INFO] [872ms] [EigenSparseLU] Solve successful, solution norm: 27746.8
[INFO] [873ms] HeatTransfer converged: iter=1 res=4.8351e-11
[INFO] [873ms] Coupling iteration 2, residual = 0.000920725
[INFO] [1.71s] [EigenSparseLU] Solve successful, solution norm: 0.00237386
[INFO] [1.71s] StructuralSolver: displacement norm = 0.00237386
[INFO] [1.71s] Coupling converged in 2 iterations
[INFO] [1.71s] Potential range: [0, 0.02] V
[INFO] [1.71s] Temperature range: [323.421, 331.43] K
[INFO] [1.71s] Temperature range: [50.2706, 58.2795] C
[INFO] [1.71s] Max displacement magnitude: 5.28686e-05 m
[INFO] [1.76s] Exported VTU results to results/busbar_results.vtu
[INFO] [1.76s] Results exported to: results/busbar_results.vtu
[INFO] [1.80s] Exported results to results/mpfem_result.txt
[INFO] [1.80s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [1.81s] === Example completed successfully! ===
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

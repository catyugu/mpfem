依据@prompt.md ，学习external/mfem和external/hpc-fem-playground中的设计
* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 我的建议是，在底层完全复现且只复现hpfem中用到的mfem的功能，不要创造太多冗余的接口或者抽象。
* 拆分BilinearForm/BoundaryBilinearForm，LinearForm/BoundaryLinearForm的基类，避免产生冗余的虚函数。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 确保对二阶问题仍然能保证求解准确性和效率。
* 重构中，不要考虑向后兼容性。
* 禁止使用mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。

```text
 HUAWEI    mpfem  new ≡  ?1   31.985s⠀   .\build\examples\busbar_example.exe                                                  pwsh   100  16:40:03 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: cases/busbar
[INFO] [0ms] Reading case from cases/busbar/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [1ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [97ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [98ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [98ms] Reading materials from cases/busbar/material.xml
[INFO] [99ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [99ms] Building electrostatics solver, order = 1
[INFO] [99ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [99ms] Domain 2 (mat2): sigma = 740700
[INFO] [100ms] Domain 3 (mat2): sigma = 740700
[INFO] [100ms] Domain 4 (mat2): sigma = 740700
[INFO] [100ms] Domain 5 (mat2): sigma = 740700
[INFO] [100ms] Domain 6 (mat2): sigma = 740700
[INFO] [100ms] Domain 7 (mat2): sigma = 740700
[INFO] [114ms] ElectrostaticsSolver: 7340 DOFs
[INFO] [115ms] Building heat transfer solver, order = 1
[INFO] [129ms] HeatTransferSolver: 7340 DOFs
[INFO] [129ms] Running coupled electro-thermal solve...
[INFO] [285ms] [EigenSparseLU] Solve successful, solution norm: 0.622902
[INFO] [285ms] Electrostatics converged: iter=1 res=3.78248e-13
[INFO] [643ms] [EigenSparseLU] Solve successful, solution norm: 27668.6
[INFO] [643ms] HeatTransfer converged: iter=1 res=4.70591e-11
[INFO] [798ms] [EigenSparseLU] Solve successful, solution norm: 0.622902
[INFO] [798ms] Electrostatics converged: iter=1 res=3.78248e-13
[INFO] [1.15s] [EigenSparseLU] Solve successful, solution norm: 27668.6
[INFO] [1.15s] HeatTransfer converged: iter=1 res=4.70591e-11
[INFO] [1.15s] Coupling converged in 2 iterations
[INFO] [1.15s] Potential range: [0, 0.02] V
[INFO] [1.15s] Temperature range: [322.497, 330.674] K
[INFO] [1.15s] Temperature range: [49.3474, 57.5238] C
[INFO] [1.18s] Exported VTU results to results/busbar_results.vtu
[INFO] [1.18s] Results exported to: results/busbar_results.vtu
[INFO] [1.18s] === Example completed successfully! ===
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
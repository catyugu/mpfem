# 大纲

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
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: cases/busbar
[INFO] [0ms] Reading case from cases/busbar/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [2ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [101ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [102ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [102ms] Reading materials from cases/busbar/material.xml
[INFO] [103ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [103ms] Building electrostatics solver, order = 1
[INFO] [103ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [103ms] Domain 2 (mat2): sigma = 740700
[INFO] [104ms] Domain 3 (mat2): sigma = 740700
[INFO] [104ms] Domain 4 (mat2): sigma = 740700
[INFO] [104ms] Domain 5 (mat2): sigma = 740700
[INFO] [104ms] Domain 6 (mat2): sigma = 740700
[INFO] [104ms] Domain 7 (mat2): sigma = 740700
[INFO] [119ms] ElectrostaticsSolver: 7340 DOFs
[INFO] [119ms] Building heat transfer solver, order = 1
[INFO] [135ms] HeatTransferSolver: 7340 DOFs
[INFO] [136ms] Running coupled electro-thermal solve...
[INFO] [219ms] [EigenSparseLU] Solve successful, solution norm: 0.622902
[INFO] [219ms] Electrostatics converged: iter=1 res=3.81512e-13
[INFO] [426ms] [EigenSparseLU] Solve successful, solution norm: 27668.6
[INFO] [426ms] HeatTransfer converged: iter=1 res=4.76993e-11
[INFO] [501ms] [EigenSparseLU] Solve successful, solution norm: 0.622902
[INFO] [501ms] Electrostatics converged: iter=1 res=3.80459e-13
[INFO] [700ms] [EigenSparseLU] Solve successful, solution norm: 27668.6
[INFO] [700ms] HeatTransfer converged: iter=1 res=4.759e-11
[INFO] [700ms] Coupling converged in 2 iterations
[INFO] [700ms] Potential range: [0, 0.02] V
[INFO] [700ms] Temperature range: [322.497, 330.674] K
[INFO] [701ms] Temperature range: [49.3474, 57.5238] C
[INFO] [741ms] Exported VTU results to results/busbar_results.vtu
[INFO] [742ms] Results exported to: results/busbar_results.vtu
[INFO] [775ms] Exported results to results/mpfem_result.txt
[INFO] [775ms] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [776ms] === Example completed successfully! ===
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
V       7.577947e-05    1.370374e-04    1.755934e-02    1.042270e-02
T       9.161004e-01    9.664351e-01    2.986729e-03    2.836634e-03
disp    2.771064e-05    5.287332e-05    5.287332e+11    0.000000e+00
```

## 需修复

* 对于设置了linzRes的一号区域，需要考虑其电导率随温度变化。
* 耦合场例如JouleHeating必须相对独立，以添加耦合项的形式加入总求解流程，而非直接把焦耳热等效应注入单场求解器中，单场求解器只应该包含自己的边界条件设置接口、源项设置接口等。
* 对于依赖于其他场的材料参数，请想办法进行中心化的管理，避免为了加入一个依赖到处拉屎，单场的定义绝对不应该包含其他场的东西。
* 编译运行，并验证准确性。
* 删除冗余的调试信息、冗余接口等；
* 此外，很多场景下，如我们案例中的场景Coefficient值、边界值等是和域/边界编号有关的，并不是一个物理场只有一个，我认为你应该修改所有物理场的接口，每一个Coefficient应该对应一组域/边界选择；
* 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。

## 修复之后
* 改写FESpace中多维度H1单元的定义，使得自由度按照节点排列，即【节点1的第1个分类，……节点1的第vdim个分量，节点2的第1个分类，……节点2的第vdim个分量，……】这样排列，以便后续的操作。
* 完成上述全部任务后，完成热膨胀耦合下的应力场计算，计算位移并进行比较。
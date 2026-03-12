依据@prompt.md ，学习external/mfem和external/hpc-fem-playground中的设计
* 仔细审查代码，关注架构问题（例如职责分离不清晰，代码不必要的冗长，模块间循环依赖，是否利于灵活处理未来除了热、电、力之外的更多耦合场等），同时减少编译依赖地狱，必要时将头文件中实现转移到同名源文件中；
* 聚焦可能的性能瓶颈，尽可能向量化，能用向量/矩阵运算的就不要手动循环，以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 参考mfem中的设计，尽可能提高组装效率，当前目标是静电场单场assemble时间小于30ms。 
* 我们现在的性能太差了…………为什么？真的该优化了！

```text
[INFO] [0ms] === Busbar Electrostatics Example ===
[INFO] [0ms] Case directory: cases/busbar
[INFO] [0ms] Reading case from cases/busbar/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [1ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [1ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [92ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [93ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [93ms] Reading materials from cases/busbar/material.xml
[INFO] [93ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [94ms] Building electrostatics solver, order = 1
[INFO] [94ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [94ms] Domain 2 (mat2): sigma = 740700
[INFO] [94ms] Domain 3 (mat2): sigma = 740700
[INFO] [94ms] Domain 4 (mat2): sigma = 740700
[INFO] [94ms] Domain 5 (mat2): sigma = 740700
[INFO] [94ms] Domain 6 (mat2): sigma = 740700
[INFO] [94ms] Domain 7 (mat2): sigma = 740700
[INFO] [100ms] Building mesh topology...
[INFO] [207ms] Topology built: 8378 boundary faces, 57853 interior faces, 9138 boundary elements mapped
[INFO] [207ms] ElectrostaticsSolver initialized: 7340 DOFs
[INFO] [207ms] BC: voltage = 0.02 V on boundary 43
[INFO] [207ms] BC: voltage = 0 V on boundary 8
[INFO] [208ms] BC: voltage = 0 V on boundary 15
[WARN] [208ms] Unknown physics kind: heat_transfer
[WARN] [208ms] Unknown physics kind: solid_mechanics
[INFO] [208ms] Assembling system...
[INFO] [2.14s] Applying Dirichlet BCs: 3 boundary conditions defined
[INFO] [2.14s] Processing boundary ID: 8
[INFO] [2.14s] Found 88 boundary elements with ID 8
[INFO] [2.14s] Processing boundary ID: 15
[INFO] [2.14s] Found 96 boundary elements with ID 15
[INFO] [2.14s] Processing boundary ID: 43
[INFO] [2.14s] Found 104 boundary elements with ID 43
[INFO] [2.14s] Applying elimination to 177 DOFs
[INFO] [2.14s]   DOF 890 = 0.02 V
[INFO] [2.14s]   DOF 891 = 0.02 V
[INFO] [2.14s]   DOF 893 = 0.02 V
[INFO] [2.14s]   DOF 971 = 0.02 V
[INFO] [2.14s]   DOF 972 = 0.02 V
[INFO] [2.15s] Applied 177 Dirichlet BCs
[INFO] [2.15s] Solving...
[INFO] [3.58s] [EigenSparseLU] Solve successful, solution norm: 0.622902
[INFO] [3.58s] ElectrostaticsSolver converged in 1 iterations, residual = 3.78248e-13
[INFO] [3.58s] === Results ===
[INFO] [3.58s] Potential range: [0, 0.02] V
[INFO] [3.58s] Expected range: [0, 0.02] V
[INFO] [3.58s] Potential range is correct!
[INFO] [3.58s] Solver iterations: 1
[INFO] [3.58s] Solver residual: 3.78248e-13
[INFO] [3.61s] Exported VTU results to results/busbar_electrostatics.vtu
[INFO] [3.62s] Results exported to: results/busbar_electrostatics.vtu
[INFO] [3.62s] === Example completed successfully! ===
```
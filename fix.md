依据@prompt.md ，学习external/mfem和external/hpc-fem-playground中的设计
* 仔细审查代码，关注架构问题（例如职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，能用向量/矩阵运算的就不要手动循环，以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 参考mfem中的设计，尽可能提高组装效率，当前目标是静电场单场assemble时间小于30ms，热场（加上耦合效应计算）的assemble时间小于100ms。 
* 我们现在的性能太差了…………为什么？真的该优化了！

```text
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [0ms] Case directory: cases/busbar
[INFO] [0ms] Reading case from cases/busbar/case.xml
[INFO] [1ms] Loaded case definition: busbar with 3 physics fields
[INFO] [2ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [2ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [132ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [134ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [134ms] Reading materials from cases/busbar/material.xml
[INFO] [135ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [135ms] Building electrostatics solver, order = 1
[INFO] [135ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [135ms] Domain 2 (mat2): sigma = 740700
[INFO] [135ms] Domain 3 (mat2): sigma = 740700
[INFO] [136ms] Domain 4 (mat2): sigma = 740700
[INFO] [136ms] Domain 5 (mat2): sigma = 740700
[INFO] [136ms] Domain 6 (mat2): sigma = 740700
[INFO] [136ms] Domain 7 (mat2): sigma = 740700
[INFO] [142ms] Building mesh topology...
[INFO] [289ms] Topology built: 8378 boundary faces, 57853 interior faces, 9138 boundary elements mapped
[INFO] [289ms] ElectrostaticsSolver initialized: 7340 DOFs
[INFO] [289ms] BC: voltage = 0.02 V on boundary 43
[INFO] [290ms] BC: voltage = 0 V on boundary 8
[INFO] [290ms] BC: voltage = 0 V on boundary 15
[INFO] [290ms] Building heat transfer solver, order = 1
[INFO] [290ms] Domain 1 (mat1): k = 400
[INFO] [291ms] Domain 2 (mat2): k = 7.5
[INFO] [291ms] Domain 3 (mat2): k = 7.5
[INFO] [291ms] Domain 4 (mat2): k = 7.5
[INFO] [291ms] Domain 5 (mat2): k = 7.5
[INFO] [292ms] Domain 6 (mat2): k = 7.5
[INFO] [292ms] Domain 7 (mat2): k = 7.5
[INFO] [298ms] HeatTransferSolver initialized: 7340 DOFs, initial T = 293.15 K
[INFO] [299ms] BC: convection h=5 W/(m虏K), Tinf=293.15 K on boundaries 1, 2, 3, 4, 5, 6, 7
[INFO] [299ms] BC: convection h=5 W/(m虏K), Tinf=293.15 K on boundaries 9, 10, 11, 12, 13, 14
[INFO] [299ms] BC: convection h=5 W/(m虏K), Tinf=293.15 K on boundaries 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
[INFO] [300ms] BC: convection h=5 W/(m虏K), Tinf=293.15 K on boundaries 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
[WARN] [300ms] Unknown physics kind: solid_mechanics
[INFO] [300ms] Coupling manager created: method=Picard, max_iter=20, tol=0.01
[INFO] [300ms] Running coupled electro-thermal solve...
[INFO] [301ms] Step 1: Solving electrostatics with constant conductivity...
[INFO] [853ms] Applying Dirichlet BCs: 3 boundary conditions defined
[INFO] [853ms] Processing boundary ID: 8
[INFO] [854ms] Found 88 boundary elements with ID 8
[INFO] [854ms] Processing boundary ID: 15
[INFO] [855ms] Found 96 boundary elements with ID 15
[INFO] [855ms] Processing boundary ID: 43
[INFO] [856ms] Found 104 boundary elements with ID 43
[INFO] [856ms] Applying elimination to 177 DOFs
[INFO] [856ms]   DOF 890 = 0.02 V
[INFO] [857ms]   DOF 891 = 0.02 V
[INFO] [857ms]   DOF 893 = 0.02 V
[INFO] [857ms]   DOF 971 = 0.02 V
[INFO] [857ms]   DOF 972 = 0.02 V
[INFO] [862ms] Applied 177 Dirichlet BCs
[INFO] [2.57s] [EigenSparseLU] Solve successful, solution norm: 0.623237
[INFO] [2.58s] ElectrostaticsSolver converged in 1 iterations, residual = 3.76255e-13
[INFO] [2.58s] Step 2: Solving heat transfer with Joule heating...
[INFO] [2.58s] HeatTransfer::assemble() - clearing previous assembly...
[INFO] [2.58s] HeatTransfer::assemble() - adding diffusion integrator...
[INFO] [2.58s] HeatTransfer::assemble() - adding convection BCs...
[INFO] [2.58s] HeatTransfer: assembling bilinear form...
[INFO] [3.25s] HeatTransfer: matrix assembled, size=7340
[INFO] [3.25s] HeatTransfer: adding heat source integrator...
[INFO] [3.25s] HeatTransfer: assembling linear form...
[INFO] [3.61s] RHS norm: 0.544856
[INFO] [3.61s] HeatTransfer: applying Dirichlet BCs...
[INFO] [3.62s] HeatTransfer: finalizing matrix...
[INFO] [3.62s] HeatTransferSolver assembled: matrix 7340x7340
[INFO] [5.40s] [EigenSparseLU] Solve successful, solution norm: 25115.3
[INFO] [5.40s] HeatTransferSolver converged in 1 iterations, residual = 4.57291e-11
[INFO] [5.40s] Temperature range: [293.15, 293.15] K
[INFO] [5.40s] Step 3: Running coupled iteration...
[INFO] [5.40s] Starting coupled electro-thermal solve (method: Picard)
[INFO] [6.27s] Applying Dirichlet BCs: 3 boundary conditions defined
[INFO] [6.27s] Processing boundary ID: 8
[INFO] [6.27s] Found 88 boundary elements with ID 8
[INFO] [6.27s] Processing boundary ID: 15
[INFO] [6.27s] Found 96 boundary elements with ID 15
[INFO] [6.27s] Processing boundary ID: 43
[INFO] [6.27s] Found 104 boundary elements with ID 43
[INFO] [6.27s] Applying elimination to 177 DOFs
[INFO] [6.27s]   DOF 890 = 0.02 V
[INFO] [6.27s]   DOF 891 = 0.02 V
[INFO] [6.27s]   DOF 893 = 0.02 V
[INFO] [6.27s]   DOF 971 = 0.02 V
[INFO] [6.27s]   DOF 972 = 0.02 V
[INFO] [6.28s] Applied 177 Dirichlet BCs
[INFO] [7.97s] [EigenSparseLU] Solve successful, solution norm: 0.623237
[INFO] [7.97s] ElectrostaticsSolver converged in 1 iterations, residual = 3.77204e-13
[INFO] [7.97s] HeatTransfer::assemble() - clearing previous assembly...
[INFO] [7.97s] HeatTransfer::assemble() - adding diffusion integrator...
[INFO] [7.98s] HeatTransfer::assemble() - adding convection BCs...
[INFO] [7.98s] HeatTransfer: assembling bilinear form...
[INFO] [9.13s] HeatTransfer: matrix assembled, size=7340
[INFO] [9.13s] HeatTransfer: adding heat source integrator...
[INFO] [9.14s] HeatTransfer: assembling linear form...
[INFO] [10.06s] RHS norm: 1.11723
[INFO] [10.06s] HeatTransfer: applying Dirichlet BCs...
[INFO] [10.07s] HeatTransfer: finalizing matrix...
[INFO] [10.07s] HeatTransferSolver assembled: matrix 7340x7340
[INFO] [12.02s] [EigenSparseLU] Solve successful, solution norm: 26390.7
[INFO] [12.02s] HeatTransferSolver converged in 1 iterations, residual = 4.67263e-11
[INFO] [12.02s] Coupling iteration 1, error = 4.835842e-02
[INFO] [13.31s] Applying Dirichlet BCs: 3 boundary conditions defined
[INFO] [13.31s] Processing boundary ID: 8
[INFO] [13.31s] Found 88 boundary elements with ID 8
[INFO] [13.31s] Processing boundary ID: 15
[INFO] [13.31s] Found 96 boundary elements with ID 15
[INFO] [13.31s] Processing boundary ID: 43
[INFO] [13.31s] Found 104 boundary elements with ID 43
[INFO] [13.31s] Applying elimination to 177 DOFs
[INFO] [13.31s]   DOF 890 = 0.02 V
[INFO] [13.31s]   DOF 891 = 0.02 V
[INFO] [13.31s]   DOF 893 = 0.02 V
[INFO] [13.31s]   DOF 971 = 0.02 V
[INFO] [13.31s]   DOF 972 = 0.02 V
[INFO] [13.32s] Applied 177 Dirichlet BCs
[INFO] [14.99s] [EigenSparseLU] Solve successful, solution norm: 0.624866
[INFO] [14.99s] ElectrostaticsSolver converged in 1 iterations, residual = 3.6955e-13
[INFO] [14.99s] HeatTransfer::assemble() - clearing previous assembly...
[INFO] [14.99s] HeatTransfer::assemble() - adding diffusion integrator...
[INFO] [14.99s] HeatTransfer::assemble() - adding convection BCs...
[INFO] [14.99s] HeatTransfer: assembling bilinear form...
[INFO] [16.62s] HeatTransfer: matrix assembled, size=7340
[INFO] [16.62s] HeatTransfer: adding heat source integrator...
[INFO] [16.63s] HeatTransfer: assembling linear form...
[INFO] [18.16s] RHS norm: 1.69748
[INFO] [18.16s] HeatTransfer: applying Dirichlet BCs...
[INFO] [18.16s] HeatTransfer: finalizing matrix...
[INFO] [18.16s] HeatTransferSolver assembled: matrix 7340x7340
[INFO] [19.82s] [EigenSparseLU] Solve successful, solution norm: 26808
[INFO] [19.82s] HeatTransferSolver converged in 1 iterations, residual = 4.20116e-11
[INFO] [19.82s] Coupling iteration 2, error = 1.557463e-02
[INFO] [21.19s] Applying Dirichlet BCs: 3 boundary conditions defined
[INFO] [21.19s] Processing boundary ID: 8
[INFO] [21.19s] Found 88 boundary elements with ID 8
[INFO] [21.19s] Processing boundary ID: 15
[INFO] [21.19s] Found 96 boundary elements with ID 15
[INFO] [21.19s] Processing boundary ID: 43
[INFO] [21.19s] Found 104 boundary elements with ID 43
[INFO] [21.19s] Applying elimination to 177 DOFs
[INFO] [21.19s]   DOF 890 = 0.02 V
[INFO] [21.19s]   DOF 891 = 0.02 V
[INFO] [21.19s]   DOF 893 = 0.02 V
[INFO] [21.19s]   DOF 971 = 0.02 V
[INFO] [21.19s]   DOF 972 = 0.02 V
[INFO] [21.20s] Applied 177 Dirichlet BCs
[INFO] [23.20s] [EigenSparseLU] Solve successful, solution norm: 0.625397
[INFO] [23.20s] ElectrostaticsSolver converged in 1 iterations, residual = 3.63175e-13
[INFO] [23.20s] HeatTransfer::assemble() - clearing previous assembly...
[INFO] [23.20s] HeatTransfer::assemble() - adding diffusion integrator...
[INFO] [23.20s] HeatTransfer::assemble() - adding convection BCs...
[INFO] [23.20s] HeatTransfer: assembling bilinear form...
[INFO] [25.61s] HeatTransfer: matrix assembled, size=7340
[INFO] [25.61s] HeatTransfer: adding heat source integrator...
[INFO] [25.61s] HeatTransfer: assembling linear form...
[INFO] [28.03s] RHS norm: 2.27962
[INFO] [28.03s] HeatTransfer: applying Dirichlet BCs...
[INFO] [28.03s] HeatTransfer: finalizing matrix...
[INFO] [28.03s] HeatTransferSolver assembled: matrix 7340x7340
[INFO] [29.73s] [EigenSparseLU] Solve successful, solution norm: 27016.7
[INFO] [29.73s] HeatTransferSolver converged in 1 iterations, residual = 4.66421e-11
[INFO] [29.73s] Coupling iteration 3, error = 7.729820e-03
[INFO] [29.73s] Coupling converged after 3 iterations
[INFO] [29.73s] Coupling converged in 3 iterations
[INFO] [29.73s] Potential range: [0, 0.02] V
[INFO] [29.73s] Temperature range: [315.008, 321.052] K
[INFO] [29.73s] Temperature range: [41.8581, 47.9024] 掳C
[INFO] [29.78s] Exported VTU results to results/busbar_results.vtu
[INFO] [29.79s] Results exported to: results/busbar_results.vtu
[INFO] [29.79s] === Example completed successfully! ===
```

* 求解正确性也可疑，在相同的设置下我们的精度应该达到这个级别：

```text
python3 scripts/compare_comsol_results.py cases/busbar/result.txt res
ults/busbar/mpfem_result.txt
field   L2      Linf    max_relative    L2_relative
V       2.148742e-09    5.572466e-08    5.224217e-06    2.965192e-07
T       2.364130e-06    7.245394e-05    2.217743e-07    7.310244e-09
disp    9.269475e-09    3.351310e-08    9.775915e-03    3.403081e-04
```
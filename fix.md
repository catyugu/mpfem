# 大纲

依据@prompt.md ，适当参考学习external/mfem和external/hpc-fem-playground中的设计

## 需求

* 仔细审查代码，关注架构问题（变量所有权管理混乱，职责分离不清晰，代码不必要的冗长，模块间循环依赖，形成依赖地狱，编译极其缓慢）。
* 聚焦可能的性能瓶颈，尽可能向量化，静态化。以及减少内存分配，同时关注底层如参考单元、形函数等模块中不必要的内存分配行为。
* 所有同质功能的接口只需要保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 重构中，不要考虑向后兼容性。
* 禁止使用const_cast, friend, mutable, dynamic_cast等关键字。
* 删除冗余的成员变量、接口等。
* 完成一块工作任务后：
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的的接口，防止冗余。
  * 提交一次代码，然后继续下一块工作。

## 工作任务1

* 目前CMakeLists.txt冗长地集中在一个文件中，请你将其模块化，例如依赖引入，各个模块的编译和依序链接等。
* MKL Eigen加速会遇到链接错误，你可能需要添加shim函数欺骗编译器来解决：

```bash
FAILED: [code=1] tests/mpfem_test_case_xml_reader.exe
C:\Windows\system32\cmd.exe /C "cd . && E:\env\cpp\msys2\clang64\bin\c++.exe -O3 -DNDEBUG  tests/CMakeFiles/mpfem_test_case_xml_reader.dir/test_case_xml_reader.cpp.obj -o tests\mpfem_test_case_xml_reader.exe -Wl,--out-implib,tests\libmpfem_test_case_xml_reader.dll.a -Wl,--major-image-version,0,--minor-image-version,0  libmpfem_io.a  libmpfem_mesh.a  lib/libgtest_main.a  _deps/tinyxml2-build/libtinyxml2.a  libmpfem_core.a  E:/env/cpp/msys2/clang64/lib/libomp.dll.a  E:/env/cpp/intel/oneAPI/2025.3/lib/mkl_intel_ilp64_dll.lib  E:/env/cpp/intel/oneAPI/2025.3/lib/mkl_intel_thread_dll.lib  E:/env/cpp/intel/oneAPI/2025.3/lib/mkl_core_dll.lib  E:/env/cpp/intel/oneAPI/2025.3/lib/libiomp5md.lib  E:/env/cpp/msys2/clang64/lib/libsuperlu.dll.a  lib/libgtest.a  -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 && cd ."
ld.lld: error: undefined symbol: __security_cookie
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_fopen)
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_fopen)
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_printf_s)
>>> referenced 21 more times

ld.lld: error: undefined symbol: __security_check_cookie
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_fopen)
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_printf_s)
>>> referenced by mkl_intel_ilp64_dll.lib(mkl_libc.obj):(mkl_serv_fprintf_s)
>>> referenced 9 more times
c++: error: linker command failed with exit code 1 (use -v to see invocation)
[5/19] Linking CXX static library libmpfem_fe.a
ninja: build stopped: subcommand failed.
```

* 正确地添加SuiteSparse的求解器。注意线性求解器配置不应该有fallback逻辑，如果不支持则在运行时直接抛出异常！！
* Eigen, SuiteSparse，OpenBLAS等已经正确安装，如果还有其他需要，可以通过MSYS2 CLANG64的pacman命令安装依赖。
* 现在求解器工厂的管理太混乱了，有很多同质接口，字符串名字、枚举类型混用，请采取一个统一的结构化求解器配置入口。

## 工作任务2

* CoupledManager灵活性有限。
* 焦耳热添加了单独的系数src/coupling/joule_heating.hpp，而热膨胀则没有，缺乏一致性。要么两个都单独配置，要么两个都不要。
* 很多场景下，如我们案例中的材料Coefficient值、边界值等是和域/边界编号有关的，并不是一个物理场只有一个，我认为你应该修改所有物理场的接口，每一个材料值应该应该对应一组域/边界选择，而不是一个求解器只持有一个或者两个特定的Coefficient；此外，应该强调，物理场持有的应该是Coefficient的基类，而非什么派生类。
* 例如，setConductivity, conductivity等接口应该指定域选择。
* 考虑以后扩展到求解瞬态问题，以及更多参数耦合问题的需求，对代码做必要的重构和抽象。例如Coefficient的eval接口应该有时间参数t。

## 工作任务3

* 调查精度问题和比较显著的性能问题，进一步优化代码。
* 当前运行结果

```text
$ ./build/examples/busbar_example.exe 
[INFO] [0ms] === Busbar Electro-Thermal Example ===
[INFO] [2ms] Case directory: cases/busbar
[INFO] [3ms] Reading case from cases/busbar/case.xml
[INFO] [4ms] Loaded case definition: busbar with 3 physics fields
[INFO] [5ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [6ms] Reading mesh from cases/busbar/mesh.mphtxt
[INFO] [61ms] Mesh loaded: 7340 vertices, 31021 volume elements, 9138 boundary elements
[INFO] [132ms] Boundary mapping: 8378 external, 760 internal (will skip in BC)
[INFO] [134ms] Mesh loaded: 7340 vertices, 31021 elements
[INFO] [134ms] Reading materials from cases/busbar/material.xml
[INFO] [135ms] Loaded 2 materials from cases/busbar/material.xml
[INFO] [136ms] Building electrostatics solver, order = 1
[INFO] [137ms] Domain 1 (mat1): sigma = 5.998e+07
[INFO] [138ms] Domain 2 (mat2): sigma = 740700
[INFO] [139ms] Domain 3 (mat2): sigma = 740700
[INFO] [139ms] Domain 4 (mat2): sigma = 740700
[INFO] [139ms] Domain 5 (mat2): sigma = 740700
[INFO] [140ms] Domain 6 (mat2): sigma = 740700
[INFO] [140ms] Domain 7 (mat2): sigma = 740700
[INFO] [164ms] ElectrostaticsSolver: 7340 DOFs
[INFO] [165ms] Building heat transfer solver, order = 1
[INFO] [184ms] HeatTransferSolver: 7340 DOFs
[INFO] [185ms] Building structural solver, order = 1
[INFO] [186ms] Domain 1 (mat1): E = 110, nu = 0.35, alpha_T = 1.7e-05
[INFO] [187ms] Domain 2 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [188ms] Domain 3 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [189ms] Domain 4 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [191ms] Domain 5 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [191ms] Domain 6 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [192ms] Domain 7 (mat2): E = 105, nu = 0.33, alpha_T = 7.06e-06
[INFO] [321ms] StructuralSolver: 22020 DOFs
[INFO] [321ms] Joule heating domains: 7 domains
[INFO] [321ms] Thermal expansion coupling enabled
[INFO] [321ms] Domain 1 (mat1): temp-dep sigma, rho0 = 1.72e-08, alpha = 0.0039
[INFO] [322ms] Running coupled electro-thermal solve...
[INFO] [341ms] Electrostatics assemble completed in 0.018s
[INFO] [386ms] [SuperLU] Solve successful, solution norm: 0.623237
[INFO] [388ms] Linear solve (SuperLU) completed in 0.046s
[INFO] [389ms] Electrostatics converged: iter=1 res=0
[INFO] [453ms] HeatTransfer assemble completed in 0.063s
[INFO] [505ms] [SuperLU] Solve successful, solution norm: 27772.3
[INFO] [507ms] Linear solve (SuperLU) completed in 0.053s
[INFO] [508ms] HeatTransfer converged: iter=1 res=0
[INFO] [508ms] Coupling iteration 1, residual = 1
[INFO] [525ms] Electrostatics assemble completed in 0.016s
[INFO] [578ms] [SuperLU] Solve successful, solution norm: 0.626625
[INFO] [580ms] Linear solve (SuperLU) completed in 0.054s
[INFO] [582ms] Electrostatics converged: iter=1 res=0
[INFO] [645ms] HeatTransfer assemble completed in 0.062s
[INFO] [695ms] [SuperLU] Solve successful, solution norm: 27746.8
[INFO] [697ms] Linear solve (SuperLU) completed in 0.051s
[INFO] [699ms] HeatTransfer converged: iter=1 res=0
[INFO] [700ms] Coupling iteration 2, residual = 0.000920725
[INFO] [780ms] Structural assemble completed in 0.079s
[INFO] [1.25s] [SuperLU] Solve successful, solution norm: 0.00237386
[INFO] [1.26s] Linear solve (SuperLU) completed in 0.476s
[INFO] [1.26s] StructuralSolver: displacement norm = 0.00237386
[INFO] [1.26s] Coupling solve completed in 0.936s
[INFO] [1.26s] Coupling converged in 2 iterations
[INFO] [1.26s] Potential range: [0, 0.02] V
[INFO] [1.26s] Temperature range: [323.421, 331.43] K
[INFO] [1.26s] Temperature range: [50.2706, 58.2795] C
[INFO] [1.26s] Max displacement magnitude: 5.28686e-05 m
[INFO] [1.33s] Exported VTU results to results/busbar_results.vtu
[INFO] [1.33s] Results exported to: results/busbar_results.vtu
[INFO] [1.38s] Exported results to results/mpfem_result.txt
[INFO] [1.38s] COMSOL format results exported to: results/mpfem_result.txt
[INFO] [1.38s] === Example completed successfully! ===
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

当前的误差是：

```text
python .\scripts\compare_comsol_results.py ./results/mpfem_result.txt ./cases/busbar/result.txt
field   L2      Linf    max_relative    L2_relative
V       6.631425e-07    1.193437e-06    1.502601e-04    9.066653e-05
T       2.772982e-03    4.183303e-03    1.262226e-05    8.562125e-06
disp    2.466711e-09    4.695249e-09    1.109621e-04    8.902466e-05
```

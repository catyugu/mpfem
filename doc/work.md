# 大纲

## 前言

* 你必须首先阅读我们项目的README.md

* 这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 任务

根据您提供的 `mpfem` 项目源代码，我识别出以下**设计反模式、冗长、循环依赖和性能隐患**，并针对**编译/链接速度慢**这一首要痛点，给出分步重构计划。以下分析基于对 `CMakeLists.txt`、`Targets.cmake`、各模块头文件及 `src` 结构的审查。

---

## 🔍 一、关键问题诊断（编译/链接缓慢的根源）

### 1. 库依赖链存在**循环依赖与反向依赖**
- `mpfem_problem` 显式链接了 `mpfem_physics`（见 `Targets.cmake` 第 132 行），但 `mpfem_physics` 却**编译了** `src/problem/transient_problem.cpp`，导致 `physics` 库反向依赖 `problem` 层的符号。  
  → 链接时可能出现重复符号、延长链接时间，且破坏模块分层。

### 2. 大量实现细节**裸露在头文件中**
- 几乎所有模块的头文件（`.hpp`）都包含了完整的成员变量定义、内联函数实现（如 `FESpace`、`ElementTransform`、`GridFunction` 等）。  
  → 任何对私有成员的修改都会触发**大规模重新编译**，且头文件展开后体积庞大，拖慢编译速度。

### 3. 过度使用模板且**无显式实例化**
- `solver/eigen_solver.hpp` 中的 `EigenIterativeOperator` 是模板类，但未在 `.cpp` 中显式实例化常用类型（如 `CgOperator`、`GmresOperator`）。  
  → 每个包含该头文件的编译单元都会重复实例化模板代码，增加编译时间和二进制体积。

### 4. **头文件包含地狱**
- 核心头文件 `core/types.hpp` 包含了 `Eigen/Core`、`Eigen/SparseCore`，且几乎所有模块都直接或间接包含它。  
  → 修改 `types.hpp` 会导致几乎全项目重编。

### 5. **库粒度不合理**
- `mpfem_physics` 库同时包含了三个物理场求解器和时间积分器，且依赖 `assembly`、`solver`、`io` 等。  
  → 任何一个小模块的改动都可能触发 `physics` 库的重链。

### 6. **缺少前向声明**
- 例如 `mesh.hpp` 中 `#include "element.hpp"`，而 `element.hpp` 又包含 `geometry.hpp`，但很多地方仅需指针或引用即可使用前向声明。

---

## 📐 二、重构目标与原则

1. **严格分层**：消除循环依赖，形成单向依赖链。
2. **隐藏实现**：大量使用 PIMPL 手法，将私有成员移入 `.cpp`。
3. **减少模板膨胀**：显式实例化常用模板，或将模板改为运行时分发。
4. **精简头文件**：用前向声明替代 `#include`，提取纯类型定义到独立轻量头文件。
5. **库拆分与合并**：按职责重组库目标，减少链接时符号解析开销。

---

## 🛠️ 三、分步重构计划（按优先级排序）

### **Phase 1：解决循环依赖与库重组（最关键，直接改善链接速度）**

| 步骤 | 任务                                                                                                                                                                                                     | 预期效果                                       |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 1.1  | **修复 `problem` ↔ `physics` 反向依赖**：将 `src/problem/transient_problem.cpp` 移至 `mpfem_problem` 库，并移除 `mpfem_problem` 对 `mpfem_physics` 的依赖。`physics` 只应被 `problem` 依赖，而不能反向。 | 消除循环链接，减少重复符号解析。               |
| 1.2  | **拆分 `mpfem_physics` 为三个独立库**：`mpfem_electrostatics`、`mpfem_thermal`、`mpfem_structural`。每个库只依赖 `mpfem_assembly`、`mpfem_core`、`mpfem_fe`。                                            | 并行编译粒度更细，改动局部场不影响其他场链接。 |
| 1.3  | **将时间积分模块独立**：`src/time/` 编译为 `mpfem_time`，供 `mpfem_problem` 链接，而不是混入 `physics`。                                                                                                 | 时间积分修改不再触发物理场重链。               |
| 1.4  | **调整 `Targets.cmake` 依赖顺序**，确保依赖关系为：<br>`core` → `mesh` → `fe` → `expr` → `io` → `assembly` → `solver` → `time` → `physics_*` → `problem`。                                               | 清晰的单向依赖图，链接器可更快确定符号位置。   |

### **Phase 2：PIMPL 化核心类（显著减少头文件改动传播）**

优先处理以下频繁被包含且实现复杂的类：

| 类                 | 文件                       | 重构方式                                                                                                                                                          |
| ------------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Mesh`             | `mesh/mesh.hpp`            | 将 `vertices_`、`elementGeoms_` 等容器移入 `Mesh::Impl`，在 `.cpp` 中定义。对外仅保留查询接口。                                                                   |
| `FESpace`          | `fe/fe_space.hpp`          | 将 `elemDofs_`、`bdrElemDofs_` 等移入 PIMPL。                                                                                                                     |
| `ElementTransform` | `fe/element_transform.hpp` | 将 `nodesBuf_`、`jacobian_` 等缓存移入 PIMPL，因为它在热路径中被频繁创建（每个线程一个）。                                                                        |
| `GridFunction`     | `fe/grid_function.hpp`     | 仅有 `Eigen::VectorXd values_`，但包含 `fes_` 指针，PIMPL 收益一般；可保留但将 `values_` 改为 `std::vector<Real>` 以减少 Eigen 头文件包含？不，Eigen 已普遍包含。 |
| `VariableManager`  | `expr/variable_graph.hpp`  | 内部 `nodes_` 和编译状态可移入 PIMPL。                                                                                                                            |

> **实施策略**：每完成一个类的 PIMPL 化，全量编译一次，确认编译时间下降。

### **Phase 3：头文件清理与前向声明**

- 创建 `core/fwd.hpp`，集中放置所有类的前向声明（`class Mesh; class FESpace; ...`）。
- 修改各模块头文件，将 `#include "mesh/mesh.hpp"` 替换为 `#include "core/fwd.hpp"`，并在 `.cpp` 中包含完整头文件。
- 将 `core/types.hpp` 拆分为：
  - `core/types_fwd.hpp`：仅含 `Index`、`Real`、`Vector3` 等基础类型别名（不含 Eigen 实现）。
  - `core/types_eigen.hpp`：包含 Eigen 类型别名，仅供需要完整 Eigen 的模块包含。

### **Phase 4：模板显式实例化**

- 在 `solver/eigen_solver.cpp` 中显式实例化 `CgOperator` 和 `GmresOperator`：
  ```cpp
  template class EigenIterativeOperator<Eigen::ConjugateGradient<...>, CgName>;
  template class EigenIterativeOperator<Eigen::DGMRES<...>, GmresName>;
  ```
- 将 `EigenIterativeOperator` 的模板定义移至 `.cpp` 或单独的 `-inl.hpp` 文件，头文件仅保留声明。

### **Phase 5：启用预编译头（PCH）**

在 CMake 中为目标（尤其是 `mpfem_core`、`mpfem_fe`等）启用预编译头，包含最常用的：
```cpp
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <memory>
#include <string>
```
可大幅加速编译（尤其在 MSVC 下）。

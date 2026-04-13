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

根据提供的全部源代码，我对 `mpfem` 项目进行了全面审查。以下分析综合了现有文档 `doc/work.md` 的洞察，并针对 **编译/链接速度缓慢** 这一核心痛点，给出具体、可执行的步骤化重构方案。所有建议均遵循项目宪法中 **“拒绝向后兼容性、强制改写所有调用处、消除冗余”** 的原则。

---

## 🔍 一、编译缓慢的根本原因诊断（补充细节）

| 瓶颈类别             | 具体表现（代码证据）                                                                                                                                  | 影响程度 |
| :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------- |
| **头文件包含地狱**   | `core/types.hpp` 被几乎全部模块包含，而其内部引入了 `Eigen/Core`、`Eigen/SparseCore`。`mesh/mesh.hpp` 包含 `element.hpp`，后者又包含 `geometry.hpp`。 | ⭐⭐⭐⭐⭐    |
| **模板未显式实例化** | `solver/eigen_solver.hpp` 中 `EigenIterativeOperator` 是模板，每个包含该头文件的 `.cpp` 都重复编译模板代码。                                          | ⭐⭐⭐⭐     |
| **缺乏前向声明**     | 头文件中大量使用 `#include "xxx.hpp"` 而非前向声明，如 `mesh.hpp` 包含 `element.hpp`。                                                                | ⭐⭐⭐      |

---

## 🚫 二、设计反模式与代码异味清单

1. **God Class**：`FESpace::buildDofTable()` 过于庞大，逻辑混杂了拓扑遍历、DOF计数、偏移计算。
2. **Inconsistent Ownership**：`LinearOperator` 持有矩阵指针但明确不拥有所有权（借用指针），而 `SparseMatrix` 封装了 Eigen 矩阵，所有权明确。混用裸指针和 `std::unique_ptr` 易造成悬空。
3. **Overly Clever Template**：`EigenIterativeOperator` 模板参数化 SolverName 字符串，但没有带来显著收益，反而增加编译负担。
4. **Magic Number**：`MaxDofsPerElement = 81` 等常量出现在多处，部分用于栈上数组大小，但未在编译期统一计算。
5. **冗余的 `VariableNode` 层次**：`GridFunctionValueProvider`、`DomainMultiplexerProvider` 等都是继承自 `VariableNode`，每次求值都有虚函数调用开销，且每个变量节点都需要动态分配。
6. **XML 解析中的字符串拷贝**：`case_xml_reader.cpp` 中使用 `std::map` 存储参数，每次解析都产生大量临时字符串。
7. **注释掉的代码和 TODO**：`src/fe/quadrature.cpp` 中注释了部分积分点，但未清理。

---

## 🛠️ 三、步骤化重构计划（强制执行，无向后兼容）

重构将按 **编译加速效果从大到小** 排序，每阶段完成后 **必须通过全部测试用例** (`doc/validation.md`)，且 **立即删除旧接口**。

### **Phase 1：消除循环依赖**

| 步骤 | 操作 | 涉及文件 |
| :--- | :--- | :------- ||
| 1.1  | **移除 `mpfem_io` 对 `mpfem_mesh` 的 `PRIVATE` 依赖**<br> - 改用前向声明 + PIMPL 隐藏实现，使 `io` 库成为纯接口层。                                                             | `io/*.hpp`, `io/*.cpp`               |

### **Phase 2：PIMPL 化核心类（预计编译时间减少 40%）**

优先处理 **被包含频率最高** 且 **实现复杂** 的类。

| 类                 | 修改方案                                                                                                          |
| :----------------- | :---------------------------------------------------------------------------------------------------------------- |
| `Mesh`             | - 在 `mesh/mesh.cpp` 中定义 `Mesh::Impl`，移动所有数据成员（`vertices_`, `elementGeoms_`, `topologyBuilt_` 等）。 |
| `FESpace`          | - 将 `elemDofs_`, `bdrElemDofs_`, `buildDofTable()` 实现完全移入 `.cpp`。                                         |
| `ElementTransform` | - 将 `nodesBuf_`, `jacobian_`, `geoShapeDerivatives_` 等缓存移入 PIMPL。                                          |
| `ReferenceElement` | - 将预计算的 `cachedShapeValues_` 和 `cachedDerivatives_` 移入 PIMPL。                                            |
| `VariableManager`  | - 将 `nodes_` 和编译状态移入 PIMPL。                                                                              |

### **Phase 3：头文件清理与前向声明（预计编译时间再减少 25%）**

| 步骤 | 操作                                                                                                                                                                                                   |
| :--- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1  | 创建 `core/fwd.hpp`，集中放置所有类的**前向声明**（`class Mesh; class FESpace; class ElementTransform; ...`）。                                                                                        |
| 3.2  | 修改各模块头文件，将 `#include "mesh/mesh.hpp"` 替换为 `#include "core/fwd.hpp"`，并在 `.cpp` 中包含完整头文件。                                                                                       |
| 3.3  | 移除所有不必要的头文件包含，例如在 `element.hpp` 中移除 `#include "geometry.hpp"`，改用前向声明 `enum class Geometry : std::uint8_t;`（因为 `element.hpp` 只存储 `Geometry` 枚举值，不需要完整定义）。 |

### **Phase 4：模板显式实例化与内联削减（减少二进制体积和编译时间）**

| 操作                                               | 涉及文件                                                                                                                                                                                                                                          |
| :------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **显式实例化 `EigenIterativeOperator`**            | 在 `solver/eigen_solver.cpp` 中添加：<br> `template class EigenIterativeOperator<Eigen::ConjugateGradient<...>, CgName>;`<br> `template class EigenIterativeOperator<Eigen::DGMRES<...>, GmresName>;`<br> 并将模板定义移入 `-inl.hpp` 或 `.cpp`。 |
| **将 `GeometryMapping` 的静态函数实现移入 `.cpp`** | `geometry_mapping.cpp` 已存在，但 `evalShape` 等函数非常庞大，应确保所有非平凡实现均在 `.cpp` 中。                                                                                                                                                |

### **Phase 5：启用预编译头（PCH）与构建优化**

| 操作                                                                          | 说明                                                                                                                                                         |
| :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 在 `CMakeLists.txt` 中为目标（尤其是 `mpfem_core`, `mpfem_fe`）启用预编译头。 | 创建 `pch.hpp` 包含：<br> `<Eigen/Core>`<br> `<Eigen/SparseCore>`<br> `<vector>`<br> `<memory>`<br> `<string>`<br> 并在 `target_precompile_headers` 中指定。 |
| 启用并行编译与缓存                                                            | 建议使用 `ccache` 或 `sccache`，并在 CMake 中设置 `CMAKE_CXX_COMPILER_LAUNCHER`。                                                                            |

---

## 📋 四、具体代码修改示例（摘录）

#### 示例1：`Mesh` 的 PIMPL 化

```cpp
// mesh/mesh.hpp (修改后)
#ifndef MPFEM_MESH_HPP
#define MPFEM_MESH_HPP

#include "core/fwd.hpp"
#include <memory>
#include <vector>

namespace mpfem {
class Mesh {
public:
    Mesh();
    ~Mesh();  // 必须在 .cpp 中定义，因为 Impl 不完整

    // ... 公有接口保持不变 ...

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace mpfem
#endif
```

```cpp
// mesh/mesh.cpp (新增 Impl 定义)
struct Mesh::Impl {
    int dim_ = 3;
    std::vector<Vector3> vertices_;
    // ... 所有原来在头文件中的数据成员 ...
};
```

## 📌 总结

本重构计划严格遵循项目宪法中的 **“拒绝向后兼容、强制改写”** 原则，以 **编译速度** 为第一优先级，同时兼顾运行性能与代码可维护性。通过 **物理分层、隐藏实现、减少模板、前向声明** 四大手段，可使项目进入高效迭代状态。建议按 Phase 顺序逐步实施。
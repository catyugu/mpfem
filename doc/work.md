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

## 具体工作任务

针对该代码库，核心的痛点在于**违反了C++高性能计算的“零成本抽象”原则**以及**存在严重的头文件依赖地狱 (Header Dependency Hell)**。

以下是针对“编译依赖地狱”、“Vertex类冗余”、“设计反模式（高频动态分配）”等问题的步骤化重构方案：

### 步骤 1：根治编译依赖地狱与链接冗余 (Dependency Hell)
**症状**：`core/types.hpp` 是整个项目的基础头文件，但它却包含了 `<Eigen/Dense>` 和 `<Eigen/Sparse>`。这意味着项目中的**每一个文件**都会解析数十万行的Eigen高级求解器、LU分解等重型代码。此外，日志与组装器的头文件也泄露了重型依赖。
**重构动作**：
1. **替换基础线性代数依赖**：
   * 在 `core/types.hpp` 中，将 `#include <Eigen/Dense>` 替换为 `#include <Eigen/Core>`。
   * 将 `#include <Eigen/Sparse>` 替换为 `#include <Eigen/SparseCore>`。
   * 对于真正需要求解器的地方（如 `eigen_solver.hpp` 和 `sparse_matrix.hpp` 中特定的直接求解器调用），只在对应的 `.cpp` 文件中 `#include <Eigen/SparseLU>` 等。
2. **剥离 `Logger` 的 IO 依赖**：
   * `core/logger.hpp` 包含了 `<iostream>` 和 `<sstream>`（流式日志的典型反模式）。将 `LogMessage` 模板类的实现剥离到 `logger.cpp`，头文件中仅保留声明，彻底移除 IO 头文件。
3. **移除头文件中的 OpenMP 暴露**：
   * `assembly/assembler.hpp` 在头文件中包含了 `<omp.h>`。移除它。线程本地缓冲区（`ThreadBuffer`）的管理应该对外部透明，将其隐藏在 `assembler.cpp` 中（可以使用 Pimpl 惯用法或通过 `std::vector<ThreadBuffer>` 的内部初始化来规避在头文件中引入 `omp.h`）。
4. **清理循环依赖和不必要的包含**：
   * `fe/fe_space.hpp` 不应 `#include "mesh/mesh.hpp"`，只需前向声明 `class Mesh;`。
   * `expr/evaluation_context.hpp` 不应包含多余的完整定义，只需前向声明。

### 步骤 2：彻底消灭 `Vertex` 类的冗余 (Data Oriented Design)
**症状**：`mesh/vertex.hpp` 定义了 `Vertex` 类，内部不仅包装了 `std::array<Real, 3>`，还**每个顶点存储了一个 `int dim_`**，并提供 `toVector()` 方法。物理空间维度是整个 `Mesh` 统一的属性，逐个顶点存储维度不仅浪费 25% 以上的内存（考虑内存对齐），还导致缓存命中率下降，且引入了无意义的类型转换开销。
**重构动作**：
1. **删除 `mesh/vertex.hpp`**。
2. **统一使用 `Vector3`**：在 `Mesh` 类中，将 `std::vector<Vertex> vertices_;` 直接替换为 `std::vector<Vector3> vertices_;`。
3. **移除维度冗余**：消除对点维度的独立检查，空间维度统一通过 `Mesh::dim()` 访问。这消除了 `Vertex::toVector()` 的高频调用开销，实现真正的数据驱动（Data-Oriented）。

### 步骤 3：消除热点路径上的内存分配反模式 (Hot-path Heap Allocation)
**症状**：大量应在栈上或预分配内存中完成的操作，频繁返回按值拷贝的 `std::vector` 和 `Eigen::VectorXd`，导致堆内存分配器 (Heap Allocator) 成为性能瓶颈。
**重构动作**：
1. **重构 `GridFunction::getElementValues`**：
   * 当前实现：`Eigen::VectorXd getElementValues(Index elem) const`，内部创建了 `std::vector<Index>` 和 `Eigen::VectorXd` 并按值返回。
   * 修改为：`void getElementValues(Index elem, std::span<Real> outValues) const`，让调用者（如组装器）利用 `ThreadBuffer` 上的静态内存（如 `std::array`）传递缓冲，实现**零内存分配 (Zero-allocation)**。

### 步骤 4：移除 SparseMatrix 的隐式临时对象风险（向后兼容/过度设计）
**症状**：`core/sparse_matrix.hpp` 为了使用方便，重载了 `SparseMatrix operator+(const SparseMatrix& B) const` 等按值返回新矩阵的算术运算符。在 FEM 中，组装后的全局稀疏矩阵极大，意外触发一次 `A = B + C` 会瞬间产生 G 级别的内存峰值（OOM风险）。
**重构动作**：
1. 删除 `SparseMatrix operator+`，`operator-` 和 `operator*` (两矩阵相乘) 的重载。
2. 强制剥夺这种向后兼容的“语法糖”，强制开发者使用 `addScaled`、`+=` 或者 `setFromTriplets` 等就地修改 (In-place mutation) 函数，从而从编译器层面杜绝大规模临时矩阵的拷贝。

### 步骤 5：消除 `GridFunction` 的非对称接口
**症状**：`fe/grid_function.hpp` 中的 `gradient` 签名：
`Vector3 gradient(Index elem, const Vector3& xi, const Matrix3& invJacobianTranspose) const;`
它强制调用方提前计算雅可比矩阵逆转置，但 `eval` 方法却不需要。这导致调用方逻辑割裂且冗长。
**重构动作**：
将其统一修改为接收 `ElementTransform` 的引用：
`Vector3 gradient(Index elem, const ElementTransform& trans) const;`
`Real eval(Index elem, const ElementTransform& trans) const;`
因为 `ElementTransform` 内部已经通过惰性求值/预计算维护了当前积分点的 `invJacobianT()`，这样既统一了接口，又避免了反复的矩阵传参和冗余计算。
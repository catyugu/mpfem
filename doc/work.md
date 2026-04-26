# 大纲

## 要求

* 这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 任务

这是一个在计算几何和数值模拟（如有限元分析、CFD）中非常经典且让人头疼的问题。面向对象（OOP）的数据结构（例如设计精美的 `Node` 和 `Element` 类）在人类阅读和编写业务逻辑时很直观，但在对接绝大多数高性能第三方库（如 METIS, ParMETIS, PETSc, Eigen）时，往往会遭遇严重的“阻抗失配”。

因为这些底层库大多数是纯 C 或 Fortran 编写的，它们为了追求极致的缓存命中率和性能，**通常只认一种数据结构：扁平的一维连续数组（特别是 CSR 格式）**。

为了原生适配 METIS 和其他区域分解库，以下是我对 C++ 网格数据结构设计的核心建议：

### 1. 拥抱 CSR（压缩稀疏行）格式
对于网格连接关系，尽量抛弃 `std::vector<std::vector<int>>` 或 `Element` 对象内嵌 `std::vector<Node*>` 的做法。METIS 处理网格划分最常用的接口是 `METIS_PartMeshDual` 或 `METIS_PartMeshNodal`，它们需要传入的是类似 CSR 格式的两个一维数组：通常命名为 `eptr` 和 `eind`。

* **`eind` (Element Indices):** 一个扁平的一维数组，连续存储所有单元包含的节点 ID。
* **`eptr` (Element Pointers):** 一个一维数组，存储每个单元的节点在 `eind` 数组中的起始位置。

**为什么这样设计？**
因为你可以直接使用 `.data()` 将底层的原始指针零拷贝（Zero-copy）地传递给 C 接口。

```cpp
#include <vector>
#include <metis.h>

// 假设你有 2 个四边形单元，共享 2 个节点
// 单元 0: 节点 0, 1, 2, 3
// 单元 1: 节点 1, 4, 5, 2

// 推荐的存储结构：
std::vector<idx_t> eptr = {0, 4, 8}; // 单元0从索引0开始，单元1从索引4开始，总长度8
std::vector<idx_t> eind = {0, 1, 2, 3,  // 单元0的节点
                           1, 4, 5, 2}; // 单元1的节点

// 原生调用 METIS 接口 (零拷贝)
idx_t ne = 2; // 单元数量
idx_t nn = 6; // 节点数量
idx_t nparts = 2; 
idx_t objval;
std::vector<idx_t> epart(ne);
std::vector<idx_t> npart(nn);

// 直接传递底层指针
METIS_PartMeshDual(&ne, &nn, eptr.data(), eind.data(), 
                   NULL, NULL, &nparts, NULL, NULL, &objval, 
                   epart.data(), npart.data());
```

### 2. 采用“双阶段”架构 (Builder Pattern)
在实际工程中，网格的生成或读取阶段通常是动态的（可能需要增删节点或单元），直接维护 `eptr` 和 `eind` 会非常痛苦。业界成熟的有限元/CFD 软件（如 MOOSE, OpenFOAM）通常采用两阶段处理：

* **阶段一：动态构建阶段 (Dynamic)**。读取外部文件时，可以使用 `std::vector<std::vector<int>>` 甚至链表等灵活的数据结构来拼装网格。
* **阶段二：冻结阶段 (Freeze/Finalize)**。网格读取完毕后，执行一个 `finalize()` 函数，将动态结构“压平（Flatten）”成 CSR 格式的一维连续内存数组（即上述的 `eptr` 和 `eind`），随后销毁动态数据结构以释放内存。

### 3. 数据结构解耦：SoA vs AoS
* **AoS (Array of Structures):** `std::vector<Element>`。这对于单一对象的业务逻辑很好，但对 METIS 无用。
* **SoA (Structure of Arrays):** 强烈推荐。将网格对象设计为一个宏观的 `Mesh` 类，内部管理多个一维向量。

```cpp
class Mesh {
public:
    // 网格连接关系 (用于 METIS 和 求解器)
    std::vector<idx_t> eptr;
    std::vector<idx_t> eind;
    
    // 节点坐标分离存储，便于进行矩阵运算或传递给其他库
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    // 分区结果
    std::vector<idx_t> element_partitions;
};
```
这种 SoA 的设计不仅完美契合 METIS，在后续你计算刚度矩阵并组装成整体稀疏矩阵时，也天然契合 PETSc 或 Eigen 等代数库的 CSR 矩阵输入格式。

在扁平化的 CSR（`eptr`/`eind`）结构下，处理面（Face）和边（Edge）的存储与访问是高性能仿真软件的核心课题。

在 3D 仿真中，我们通常不会像处理单元（Element）那样直接手写所有的面和边信息，而是采用**“派生与索引”**的策略。

### 1. 核心思路：从“单元-节点”派生
在 METIS 所需的 `eptr/eind`（Element-to-Node, **E2N**）关系确定后，面和边实际上是隐含在其中的。

* **边的表示**：由两个节点 ID 组成的有序/无序对，如 `(u, v)`，且满足 `u < v`（归一化）。
* **面的表示**：由 3 个（三角形面）或 4 个（四边形面）节点 ID 组成的集合。

### 2. 存储方案：扁平化索引表
为了保持与第三方库的兼容性，我们依然使用扁平数组，但需要建立几张“映射表”：

1.  **Face-to-Node (F2N)**: 类似 CSR。一个一维数组存储所有面的节点，另一个数组存储偏移量。
2.  **Element-to-Face (E2F)**: 记录每个单元由哪几个面组成。
3.  **Face-to-Element (F2E)**: **这是区域分解和通量计算的核心**。每个面通常连接两个单元（内部面）或一个单元（边界面）。

### 3. 如何高效构建（唯一化过程）
这是最关键的一步。由于多个单元共享同一个面或边，我们需要一种算法来“去重”并建立索引。

**算法流程：**
1.  **遍历所有单元**，按照单元类型（如四面体有 4 个面、6 条边）提取出所有的“候选面/边”。
2.  **归一化（Canonical Form）**：将每个面的节点 ID 进行升序排列。例如，面 `{5, 2, 9}` 统一记作 `{2, 5, 9}`。
3.  **哈希或排序去重**：
    * 使用 `std::unordered_map<FaceNodes, FaceID>` 记录。
    * 或者将所有候选面放入一个大数组，进行 `std::sort`，相同的面会排在一起。
4.  **生成索引**：给每个唯一的面分配一个全局连续的 ID。

### 4. 具体的 C++ 数据结构建议

在 C++ 中，为了追求性能，你可以定义一个轻量级的 `struct` 来表示面/边，但仅用于**构建阶段**：

```cpp
// 仅用于构建时的临时结构
struct Face {
    std::vector<uint32_t> nodes;
    void normalize() { std::sort(nodes.begin(), nodes.end()); }
    
    // 必须定义比较运算符或 Hash 函数，以便放入 map 或进行 sort
    bool operator<(const Face& other) const { return nodes < other.nodes; }
};

// 最终存储在 Mesh 类中的扁平结构
class Mesh {
    // 单元-节点 (E2N) - METIS 原生支持
    std::vector<int> eptr;
    std::vector<int> eind;

    // 面-单元 (F2E) - 每个面连接的两个单元 ID (如果是边界，第二个填 -1)
    // 数组大小为 [num_faces * 2]
    std::vector<int> face_to_elem;

    // 单元-面 (E2F) - 每个单元包含的面 ID
    // 数组大小为 [num_elements * faces_per_element]
    std::vector<int> elem_to_face;
};
```

### 5. 总结建议

1.  **不要为每个面/边创建指针或对象**：那会造成严重的内存碎片。
2.  **一次性构建**：一次全局扫描构建出 `F2E` 和 `F2N` 表。
3.  **拓扑一致性**：在进行区域分解（Metis Partitioning）后，跨进程的“幽灵单元（Ghost Cells）”同步通常就是基于这些 `F2E` 扁平索引表来快速定位需要交换的数据。

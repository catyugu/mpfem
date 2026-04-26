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

你目前的 `Mesh` 类已经具备了一个现代、高性能有限元底层框架的雏形。采用 `elementOffsets_` 和 `elementNodes_` 这种纯粹的 CSR (压缩稀疏行) 结构非常棒，完全踩在了 METIS/ParMETIS 等 C 语言底层库的审美点上。同时，通过 `Element` 视图（View）模式来避免对象的拷贝，这也是 C++20 `std::span` 时代非常推荐的做法。

但如果以“工业级开源求解器”（如 MFEM, MOOSE, FEniCS）的标准来审视，目前的类依然存在“上帝对象（God Object）”的倾向，且在拓扑构建的底层存在比较严重的性能隐患。

为了让它更干净且符合工业标准，我建议从以下维度进行重构：

### 1. 致命性能 Bug 修复：消除拓扑构建时的堆分配 (Heap Allocation)
在你的 `mesh.cpp` 中，我注意到了一个在大规模网格下会导致内存碎片化和极度缓慢的设计：

```cpp
// mesh.hpp
using FaceKeyType = std::vector<Index>; 
// mesh.cpp
struct FaceCandidate {
    std::vector<Index> nodes; // 极度危险：海量微小堆分配
    Index elemIdx;
    int localFace;
};
```
在 `buildFaceToElementMap` 中，对于一个 1000 万单元的网格，这会立刻触发四五千万次 `std::vector` 的 `malloc` 堆分配，随后还要进行排序和销毁。性能会在这里出现断崖式下跌。

**工业级做法**：对于网格的“面”或“边”，其节点数是极其有限的（三角形 3 个，四边形 4 个）。绝不允许在拓扑遍历的内层循环中使用动态长度的 `std::vector`。
**建议**：引入一个栈上的短数组结构（类似 LLVM 的 `SmallVector` 或固定大小的 `std::array`），通过内联排序（Inline Sort）来作为 Hash Key：

```cpp
// 用于拓扑推导的轻量级结构，完全在栈上分配
struct FaceKey {
    Index nodes[4]; // 假设最多支持四边形面
    int count;

    // 构造时直接插入排序
    FaceKey(std::span<const Index> raw_nodes) {
        count = raw_nodes.size();
        for(int i=0; i<count; ++i) nodes[i] = raw_nodes[i];
        std::sort(nodes, nodes + count);
    }
    
    bool operator<(const FaceKey& o) const {
        if (count != o.count) return count < o.count;
        for (int i = 0; i < count; ++i) {
            if (nodes[i] != o.nodes[i]) return nodes[i] < o.nodes[i];
        }
        return false;
    }
    bool operator==(const FaceKey& o) const { ... }
};
```

### 2. 数据布局：SoA 坐标系统与 C 库接口的博弈
你提到使用了 SoA（`x_`, `y_`, `z_` 独立数组）是为了 METIS 兼容。这是一个常见的误解。
实际上，METIS 的标准图划分 (`METIS_PartMeshDual`) 完全不需要坐标。而当你升级到 **ParMETIS** 进行分布式网格几何划分（或者对接 ParaView/VTK 导出，以及 CGNS 等底层 C 格式）时，它们通常要求的是一个**交织的一维数组 (AoS 平铺)**。

**建议**：将 `x_, y_, z_` 合并为一个 `std::vector<Real> coords_;`，大小为 `numNodes * dim`。
存储格式为 `[x0, y0, z0, x1, y1, z1...]`。这样在调用 C API 时，直接传 `coords_.data()` 即可实现真正的 Zero-copy（零拷贝），而分离的 SoA 反而需要你在对接外部库时重新组装打包。
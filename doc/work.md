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

将现有的自定义有限元基函数定义（`src/fe`）迁移到成熟的外部库，是提升求解器稳定性、支持更高阶/复杂空间（如 H(curl), H(div)）以及降低长期维护成本的明智决定。

结合你的需求（标量/向量/矩阵语义、PIMPL 严格隔离、高性能、无后向兼容负担），以下是系统性重构指导和外部库选型推荐。

### 一、 核心外部库选型推荐

1.  **有限元单元定义 (FE Definitions): FEniCS Basix**
    * **理由**：Basix 是 FEniCSx 项目的基石，专门负责计算有限元基函数及其导数。它是一个纯 C++ 库，极其轻量且高性能。它原生支持 Lagrange (H1), Nédélec (ND/Hcurl), Raviart-Thomas (RT/Hdiv) 等丰富族类，且支持任意阶数和张量积单元。

### 二、 PIMPL 架构重构指南与核心代码片段

**核心原则**：
* **头文件纯净**：`.hpp` 文件中绝对不出现 `<basix/...>`。
* **ABI 隔离**：使用 `std::unique_ptr<Impl>`，且**必须**在 `.cpp` 文件中实现析构函数，否则会导致智能指针析构不完整类型的编译错误。
* **现代 C++ 数据传递**：接口层摒弃裸指针，使用 `std::span` (C++20) 进行安全且零拷贝的内存边界传递。

#### 1.  FE 单元系统重构 (`BasisEvaluator`)

抛弃原有的 `src/fe/h1.cpp`, `src/fe/nd.cpp` 等硬编码实现。利用 Basix 统一掌管形状函数（Shape Functions）及其多阶导数（Gradients, Hessians）的评估。

**basis_evaluator.hpp (纯净接口)**
```cpp
#pragma once

#include <memory>
#include <span>
#include <vector>

namespace fem::fe {

enum class ElementFamily {
    Lagrange,    // H1 空间
    Nedelec,     // H(curl) 空间
    RaviartThomas // H(div) 空间
};

enum class CellTopology {
    Triangle,
    Tetrahedron,
    Hexahedron
};

class BasisEvaluator {
public:
    BasisEvaluator(ElementFamily family, CellTopology topo, int degree);
    ~BasisEvaluator();

    BasisEvaluator(BasisEvaluator&&) noexcept;
    BasisEvaluator& operator=(BasisEvaluator&&) noexcept;

    int num_dofs() const;
    int value_dimension() const; // 标量基为1，向量基为2或3

    // 核心求值：在给定的参考坐标点上计算形状函数及其导数
    // derivative_order = 0 (仅形状函数), 1 (梯度), 2 (Hessian)
    // 输入 points: 扁平化的坐标数组 [x0, y0, z0, x1, y1, z1, ...]
    // 输出 results: 扁平化的结果张量
    void tabulate(int derivative_order, 
                  std::span<const double> points, 
                  std::vector<double>& results) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace fem::fe
```

**basis_evaluator.cpp (依赖隐藏实现)**
```cpp
#include "basis_evaluator.hpp"

// 外部库依赖仅在此处暴露
// #include <basix/finite-element.h>
// #include <basix/cell.h>
// #include <basix/polynomials.h>

namespace fem::fe {

struct BasisEvaluator::Impl {
    // basix::FiniteElement element;
    int degree;
    
    Impl(ElementFamily fam, CellTopology topo, int deg) : degree(deg) {
        // 根据枚举映射到 basix::element::family 和 basix::cell::type
        // element = basix::create_element(mapped_fam, mapped_topo, deg, ...);
    }
};

BasisEvaluator::BasisEvaluator(ElementFamily family, CellTopology topo, int degree)
    : pimpl_(std::make_unique<Impl>(family, topo, degree)) {}

BasisEvaluator::~BasisEvaluator() = default;
BasisEvaluator::BasisEvaluator(BasisEvaluator&&) noexcept = default;
BasisEvaluator& BasisEvaluator::operator=(BasisEvaluator&&) noexcept = default;

int BasisEvaluator::num_dofs() const {
    // return pimpl_->element.dim();
    return 0;
}

int BasisEvaluator::value_dimension() const {
    // return pimpl_->element.value_size();
    return 1;
}

void BasisEvaluator::tabulate(int derivative_order, 
                              std::span<const double> points, 
                              std::vector<double>& results) const {
    // std::array<std::size_t, 2> shape = { points.size() / topo_dim, topo_dim };
    // var basix_result = pimpl_->element.tabulate(derivative_order, points_data, shape);
    // 拷贝或移动 basix_result 到 results 中
}

} // namespace fem::fe
```

### 三、 构建系统 (CMake) 改造指导

为了贯彻依赖隔离的原则，在引入Basix 时，必须在 `CMakeLists.txt` 中使用 `PRIVATE` 关键字链接。这样即使其他模块（如 `src/assembly`）链接 `fe` 库，也不会感知到第三方库的头文件。

```cmake
# CMakeLists.txt 片段

# 构建 FE 模块
add_library(fem_fe STATIC 
    src/fe/basis_evaluator.cpp
    # 其他不含特定单元硬编码的基础结构
)
target_include_directories(fem_fe PUBLIC src/fe)
# 【强制要求】PRIVATE 链接外部库，阻止头文件传染
target_link_libraries(fem_fe PRIVATE basix) 

# 核心求解器
add_library(fem_core STATIC src/assembly/assembler.cpp)
target_link_libraries(fem_core PUBLIC fem_fe fem_expr)
```


## 迁移步骤

### 核心原则：建立防线

我们不应该在核心组装阶段到处打补丁，而是要在**网格读取**和**自由度分配**的边界上完成所有转换。

---

### 阶段一：对齐基础几何与顶点顺序 (Mesh Input 层)

Basix 采用的参考单元（Reference Cell）定义源自 FIAT / FEniCS / UFC 规范。对于常见的单元，它的顶点和边面编号有严格规定。

**典型差异（以六面体 Hexahedron 为例）：**
* 很多传统软件（如 ANSYS, Gmsh）的六面体顶点顺序是底面逆时针 0-1-2-3，顶面 4-5-6-7。
* **Basix 的 Hexahedron 顶点顺序是基于张量积的**：底面是 0-1-3-2，顶面是 4-5-7-6。（注意 2 和 3 的位置互换了，它是 $z \otimes y \otimes x$ 的笛卡尔积顺序）。

**重构动作：在 IO 层建立转接映射**

修改 `src/io/mphtxt_reader.cpp` 和其他网格读取器。当从文件读入单元连通性（Connectivity）时，**立刻**将其重排为 Basix 顺序，存入 `src/mesh/mesh` 中。内部网格数据结构只保留 Basix 顺序。

```cpp
// 在 IO 层或 Mesh Builder 中定义映射表
namespace fem::mesh::mapping {

// 例如：Gmsh 的 8 节点六面体到 Basix 的映射
constexpr int gmsh_to_basix_hex8[8] = {0, 1, 3, 2, 4, 5, 7, 6};

// Gmsh 的 10 节点四面体到 Basix 顺序 (高阶节点的边排序差异极大)
// Basix 四面体边顺序: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
constexpr int gmsh_to_basix_tet10[10] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 8}; 

} // namespace fem::mesh::mapping

// 在网格读取时：
void read_elements(...) {
    for (int i = 0; i < num_nodes_per_elem; ++i) {
        // 读取后立刻转换，内部 Mesh 结构彻底与 Basix 同构
        internal_connectivity[i] = raw_data[ mapping::gmsh_to_basix_hex8[i] ]; 
    }
}
```

---

### 阶段二：重写网格拓扑生成 (Mesh Topology 层)

你的 `src/mesh/mesh.cpp` 中肯定有生成边（Edge）和面（Face）的代码（用于 H(curl) 和 H(div) 单元）。你必须废弃原有的硬编码生成逻辑。

**重构动作：使用 Basix 规范重新定义局部实体连通性**

Basix 的 `basix::cell::topology(cell_type)` 返回了标准的实体定义。为了避免在网格底层直接引入 Basix 依赖，你可以对照 Basix 的规范重写你的 `src/core/geometry.hpp` 中的常量表。

以四面体 (Tetrahedron) 为例，你必须遵循 Basix 的定义：
* **Vertices**: 0, 1, 2, 3
* **Edges (6条)**: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
* **Faces (4个)**: (1,2,3), (0,2,3), (0,1,3), (0,1,2)

你的 `Mesh::build_edges_and_faces()` 函数在遍历单元顶点提取边和面时，**必须按上述确切的顶点组合和方向生成**，否则方向（Orientation）会错乱。

---

### 阶段三：彻底重构自由度布局引擎 (FE Space 层)

这是重构的核心。原来的 `src/field/fe_space.cpp` 可能假定了某种固定的 DOF 分配（例如：每个节点1个，每条边1个）。但对于高阶空间（如 P3，Nédélec 2nd kind 等），DOF 的分布非常复杂。

**重构动作：交出 DOF 布局的控制权**

利用 Basix 提供的 API 来决定如何在网格实体（顶点、边、面、体）上分配自由度。

```cpp
// 假设这是你隔离 Basix 后的 FE Space 构建核心逻辑
void FESpace::Impl::distribute_dofs(const Mesh& mesh, const BasisEvaluator& fe) {
    // 1. 获取每种几何实体的 DOF 数量
    // basix 接口: num_entity_dofs(dim, entity_index)
    // 对于均匀单元，同维度的 entity 的 DOF 数通常一样
    const int dofs_per_vertex = fe.num_entity_dofs(0, 0); 
    const int dofs_per_edge   = fe.num_entity_dofs(1, 0);
    const int dofs_per_face   = fe.num_entity_dofs(2, 0);
    const int dofs_per_cell   = fe.num_entity_dofs(3, 0);

    // 2. 为全局网格实体分配全局 DOF ID
    int global_dof_offset = 0;
    
    // 顶点 DOF
    std::vector<int> vertex_dof_start(mesh.num_vertices());
    for(int i=0; i<mesh.num_vertices(); ++i) {
        vertex_dof_start[i] = global_dof_offset;
        global_dof_offset += dofs_per_vertex;
    }

    // 边 DOF (依据 mesh 生成的符合 Basix 规范的边)
    // ... 面 DOF, 体 DOF 依此类推

    // 3. 构建 Cell 到 DOF 的映射 (Cell-DOF Map / Local-to-Global)
    for (int cell_idx = 0; cell_idx < mesh.num_cells(); ++cell_idx) {
        std::vector<int> local_dofs;
        
        // 必须严格按照 Basix 的 entity 顺序装配 local dof
        // 顺序必须是：Vertices -> Edges -> Faces -> Cell Interior
        
        // 装配点
        for (int v = 0; v < num_vertices_per_cell; ++v) {
            int global_v = mesh.cell_vertex(cell_idx, v); // 已经是 basix 顺序
            for(int d=0; d<dofs_per_vertex; ++d) local_dofs.push_back(vertex_dof_start[global_v] + d);
        }
        
        // 装配边
        for (int e = 0; e < num_edges_per_cell; ++e) {
            int global_e = mesh.cell_edge(cell_idx, e); 
            // 注意：这里需要处理边方向 (Orientation) 带来的 DOF 翻转或重排
            // 下文详述
            // ...
        }
        
        // ... 装配面和体
        
        cell_dof_map[cell_idx] = std::move(local_dofs);
    }
}
```

---

### 阶段四：征服最难的战役 —— H(curl) / H(div) 的方向变换 (Orientation & Permutation)

在 Nédélec (H(curl)) 和 Raviart-Thomas (H(div)) 空间中，全局网格中的边方向（例如从节点 3 到节点 8）可能与当前单元的局部边方向（从局部节点 1 到 0）相反。

对于最低阶单元，这只是一个正负号（$\pm 1$）的区别。但对于高阶单元，如果边上有 3 个 DOF，方向颠倒不仅意味着变号，还意味着 **DOF 的排列顺序要倒过来**。

**重构动作：使用 Basix 的 Transformation 机制**

这是 Basix 最强大的功能之一。在 `src/assembly/assembler.cpp` 获取局部单元矩阵后，或者在装配前提取形函数时，必须应用 Basix 的 `create_transformations` 或 `create_custom_transformations` 矩阵。

由于你使用了 PIMPL，你需要将“方向信息”传递给你的 Evaluator：

```cpp
// basis_evaluator.hpp (接口)
namespace fem::fe {
class BasisEvaluator {
    // ...
    // 获取单元特定的转换矩阵
    // cell_info: 包含该单元在全局网格中各边/面的方向标志 (例如 0 为同向, 1 为反向)
    void get_dof_transformations(std::span<const int> cell_info, std::vector<double>& transform_matrix) const;
};
}

// 在装配器中 (Assembler)
void assemble_cell(int cell_idx) {
    // 1. 获取基函数参考值
    evaluator.tabulate(..., ref_values);

    // 2. 根据该单元在全局网格中的拓扑方向，获取 Basix 提供的转换矩阵
    std::vector<int> cell_orientation_info = mesh.get_cell_orientation(cell_idx);
    std::vector<double> transform_mat;
    evaluator.get_dof_transformations(cell_orientation_info, transform_mat);

    // 3. 将 ref_values 与 transform_mat 相乘，得到真正在这个单元上具有正确连续性的形函数
    // physical_shape_functions = transform_mat * ref_values * push_forward_mapping
    
    // 4. 计算局部刚度矩阵 ...
}
```

### 总结重构路线图

1.  **停止当前特性的开发**，冻结版本。
2.  **重写 Mesh IO**：拿出一个四面体和一个六面体，打印它的顶点、边、面顺序。与 Basix 文档进行强制对比，在读取时写入映射数组。
3.  **删代码**：删掉你现存的所有 `src/fe/h1.cpp`, `nd.cpp`, `geometry_mapping.cpp` 中关于形函数解析表达式的硬编码。
4.  **引入 PIMPL 接口**：实现 `BasisEvaluator`。
5.  **重写 FE Space 装配**：利用 Evaluator 提供的 `num_entity_dofs` 重新分配 DOF。
6.  **调试**：先用最低阶的 Lagrange P1 跑通拉普拉斯方程，验证节点对齐。再用最高阶 Nédélec 跑通旋度方程，验证 Edge 方向转换矩阵。

这个过程本质上是将你系统中那些“隐含的、经验性的”几何知识，全部替换为 Basix 标准化的张量代数描述。阵痛期过后，你的求解器将立刻具备支持任意阶数混合有限元的能力。
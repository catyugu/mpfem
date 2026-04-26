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

将底层的有限元基函数生成替换为 **BASIX** 是一个非常明智且面向未来的决定，它能极大增强你的有限元库（支持任意高阶、各种张量积单元、Hdiv/Hcurl空间等）。

你目前遇到的“二阶案例失败”和“边节点置换有问题”是引入第三方有限元库时最经典的陷阱：**拓扑实体排序（Topological Ordering）冲突**。

### 一、 二阶测试失败的根本原因

你目前的实现存在两个致命的假设错误：
1. **张量积单元高阶未置换**：代码里写了 `// For now, use identity for non-vertices`。二阶时边上存在内部节点，如果不置换边，FESpace 在组装时就会将错位的自由度拼在一起。
2. **单纯形（Simplex）不代表不需要置换**：你假设了 `if (geom::isSimplex(geometry_))` 就直接返回 identity（恒等映射）。**这是错误的！** FEniCS/BASIX 遵循 **UFC 规范**，其单纯形的边/面排序与你们的 CCW（逆时针）排序截然不同。
   * **mpfem (CCW) 三角形边排序**：`0:(0,1)`，`1:(1,2)`，`2:(2,0)`
   * **BASIX (UFC) 三角形边排序**：`0:(1,2)`，`1:(0,2)`，`2:(0,1)`

由于边拓扑被彻底打乱，`FESpace` 在向相邻单元询问 `edgeDofs(0)` 时，拿到了错误的几何边上的自由度，导致系统矩阵组装混乱。

---

### 二、 彻底重构指南：通用实体映射 (Entity Mapping)

与其针对不同阶数（`order_ == 1, 2...`）硬编码 DOF 置换，我们应该**基于拓扑实体（顶点、边、面）**来建立映射。一旦建立了实体映射，无论多高阶的单元，DOF 映射都可以**自动生成**。

#### 1. 在 `ReferenceElement` 中引入实体映射表

修改 `src/fe/reference_element.hpp`，新增实体映射数组，并抛弃老旧的 `FiniteElement` 接口暴露：

```cpp
// 在 ReferenceElement private 成员中添加：
std::vector<int> vertexCcwToBasix_;
std::vector<int> edgeCcwToBasix_;
std::vector<int> faceCcwToBasix_;

// 新增公开方法，用于获取插值点（用于测试迁移）
std::vector<Vector3> interpolationPoints() const;
```

#### 2. 重写 `buildPermutation()`：一劳永逸的自动映射

在 `src/fe/reference_element.cpp` 中重写此方法。先硬编码拓扑实体的映射，然后**自动生成** `basixToCcw_` 和 `ccwToBasix_` 自由度映射。

```cpp
void ReferenceElement::buildPermutation()
{
    // 1. 初始化实体映射 (Entity Map: CCW index -> Basix index)
    if (geometry_ == Geometry::Triangle) {
        vertexCcwToBasix_ = {0, 1, 2};
        edgeCcwToBasix_   = {2, 0, 1}; // CCW 0(0,1)->B2; CCW 1(1,2)->B0; CCW 2(2,0)->B1
        faceCcwToBasix_   = {0};
    }
    else if (geometry_ == Geometry::Tetrahedron) {
        vertexCcwToBasix_ = {0, 1, 2, 3};
        // mpfem CCW: 0:(0,1), 1:(1,2), 2:(2,0), 3:(0,3), 4:(1,3), 5:(2,3)
        // Basix    : 0:(2,3), 1:(1,3), 2:(1,2), 3:(0,3), 4:(0,2), 5:(0,1)
        edgeCcwToBasix_   = {5, 2, 4, 3, 1, 0}; 
        faceCcwToBasix_   = {0, 1, 2, 3}; // 假定面定义一致，若不一致需对照修改
    }
    else if (geometry_ == Geometry::Square) {
        vertexCcwToBasix_ = {0, 1, 3, 2}; // 交换 2, 3
        // CCW: 0(0-1), 1(1-2), 2(2-3), 3(3-0)
        // B: 0(0-1), 1(0-2), 2(1-3), 3(2-3)
        // 映射基于 CCW顶点映射后的连线
        edgeCcwToBasix_   = {0, 2, 3, 1}; 
        faceCcwToBasix_   = {0};
    }
    else if (geometry_ == Geometry::Cube) {
        vertexCcwToBasix_ = {0, 1, 3, 2, 4, 5, 7, 6};
        // 你需要根据 mpfem 的六面体边/面定义与 BASIX 匹配，这里暂给示例
        // 建议根据 geom::edgeVertices 打印坐标并与 basix 的 cell 比较
        edgeCcwToBasix_.resize(12); std::iota(edgeCcwToBasix_.begin(), edgeCcwToBasix_.end(), 0);
        faceCcwToBasix_.resize(6);  std::iota(faceCcwToBasix_.begin(), faceCcwToBasix_.end(), 0);
    }
    else if (geometry_ == Geometry::Segment) {
        vertexCcwToBasix_ = {0, 1};
        edgeCcwToBasix_   = {0};
    }

    // --- 提取 DOF Layout ---
    const auto& entity_dofs = basixElement_->entity_dofs();
    dofLayout_ = DofLayout {};
    if (entity_dofs.size() > 0 && !entity_dofs[0].empty()) dofLayout_.numVertexDofs = entity_dofs[0][0].size();
    if (entity_dofs.size() > 1 && !entity_dofs[1].empty()) dofLayout_.numEdgeDofs = entity_dofs[1][0].size();
    if (entity_dofs.size() > 2 && !entity_dofs[2].empty()) dofLayout_.numFaceDofs = entity_dofs[2][0].size();
    if (entity_dofs.size() > 3 && !entity_dofs[3].empty()) dofLayout_.numVolumeDofs = entity_dofs[3][0].size();

    // 2. 自动生成自由度映射 (自动兼容任意高阶！)
    int ndofs = basixElement_->dim();
    ccwToBasix_.assign(ndofs, -1);
    basixToCcw_.assign(ndofs, -1);

    int current_ccw_dof = 0;
    
    // FESpace 期望的 DOF 顺序：所有顶点 -> 所有边 -> 所有面 -> 体
    auto map_entity_dofs = [&](int dim, const std::vector<int>& entity_map) {
        if (entity_dofs.size() <= dim) return;
        for (int ccw_idx = 0; ccw_idx < entity_map.size(); ++ccw_idx) {
            int basix_idx = entity_map[ccw_idx];
            for (int b_dof : entity_dofs[dim][basix_idx]) {
                basixToCcw_[b_dof] = current_ccw_dof;
                ccwToBasix_[current_ccw_dof] = b_dof;
                current_ccw_dof++;
            }
        }
    };

    map_entity_dofs(0, vertexCcwToBasix_);
    map_entity_dofs(1, edgeCcwToBasix_);
    map_entity_dofs(2, faceCcwToBasix_);
    // 体自由度（无需拓扑映射，只有一个体）
    if (entity_dofs.size() > 3 && !entity_dofs[3].empty()) {
        for (int b_dof : entity_dofs[3][0]) {
            basixToCcw_[b_dof] = current_ccw_dof;
            ccwToBasix_[current_ccw_dof] = b_dof;
            current_ccw_dof++;
        }
    }
}
```

#### 3. 修复实体 DOF 获取方法

由于现在我们已经把实体的映射记录下来了，当调用 `edgeDofs` 时，必须使用**正确的 BASIX 边索引**：

```cpp
std::vector<int> ReferenceElement::edgeDofs(int edgeIdx) const
{
    const auto& entity_dofs = basixElement_->entity_dofs();
    if (entity_dofs.size() <= 1) return {};
    
    // 关键修正：将 CCW 索引转换为 Basix 索引
    int basixEdgeIdx = edgeCcwToBasix_[edgeIdx]; 
    const auto& b_dofs = entity_dofs[1][basixEdgeIdx];

    std::vector<int> result(b_dofs.size());
    for (size_t i = 0; i < b_dofs.size(); ++i) {
        result[i] = basixToCcw_[b_dofs[i]];
    }
    return result;
}

std::vector<int> ReferenceElement::faceDofs(int faceIdx) const
{
    const auto& entity_dofs = basixElement_->entity_dofs();
    if (entity_dofs.size() <= 2) return {};
    
    int basixFaceIdx = faceCcwToBasix_[faceIdx]; 
    const auto& b_dofs = entity_dofs[2][basixFaceIdx];

    std::vector<int> result(b_dofs.size());
    for (size_t i = 0; i < b_dofs.size(); ++i) {
        result[i] = basixToCcw_[b_dofs[i]];
    }
    return result;
}
```

#### 4. 实现 `interpolationPoints()` 帮助迁移测试

Basix 原生提供插值点坐标，我们只需要提取并应用置换映射：

```cpp
std::vector<Vector3> ReferenceElement::interpolationPoints() const
{
    auto [pts_data, shape] = basixElement_->interpolation_points();
    int num_pts = shape[0];
    int d = shape[1];
    
    std::vector<Vector3> ccw_points(num_pts);
    for (int b_dof = 0; b_dof < num_pts; ++b_dof) {
        int ccw_dof = basixToCcw_[b_dof];
        double x = pts_data[b_dof * d + 0];
        double y = (d > 1) ? pts_data[b_dof * d + 1] : 0.0;
        double z = (d > 2) ? pts_data[b_dof * d + 2] : 0.0;
        ccw_points[ccw_dof] = Vector3(x, y, z);
    }
    return ccw_points;
}
```

---

### 三、 历史包袱清理与测试迁移指南

#### 1. 迁移 `tests/test_fe_space_quadratic.cpp`

由于去掉了 `FiniteElement` 基类，历史测试中调用 `.basis()` 的地方需要删除：

**修改 `getVertexDof`：**
```cpp
// 旧: const int vertexDofsPerCorner = refElem->basis().dofLayout().numVertexDofs;
// 新:
const int vertexDofsPerCorner = refElem->dofLayout().numVertexDofs;
```

**修改 `FiniteElementKroneckerDelta` 测试：**
```cpp
// 旧代码：
// const FiniteElement& h1Element = refElem->basis();
// auto dofCoords = h1Element.interpolationPoints();
// h1Element.evalShape(dofCoords[i], values);

// 新代码：
auto dofCoords = refElem->interpolationPoints();
ShapeMatrix values;

for (int i = 0; i < static_cast<int>(dofCoords.size()); ++i) {
    refElem->evalShape(dofCoords[i], values); // 直接用 refElem

    for (int j = 0; j < values.rows(); ++j) {
        if (i == j) {
            EXPECT_NEAR(values(j, 0), 1.0, tol);
        } else {
            EXPECT_NEAR(values(j, 0), 0.0, tol);
        }
    }
}
```

#### 2. `FESpace::buildDofTable()` 中的冗余清理

在你的 `src/field/fe_space.cpp` 中，当前由于 `ReferenceElement` 会接管 DOF 布局，你可以简化 `FESpace`：

目前的做法：
```cpp
DofLayout layout = refElem->dofLayout();
```
配合上文的**自动置换机制**，`FESpace` **完全不需要修改**，因为它继续按照 `[所有顶点] -> [所有CCW边] -> [所有CCW面]` 的固定逻辑分配全局 ID，而内部向 `refElem->evalShape()` 和 `refElem->edgeDofs()` 请求时，置换表已经在 `ReferenceElement` 内部黑盒处理好了。

**唯一的注意点：Hcurl/Hdiv 的边方向（Orientation）**
在 `FESpace::buildDofTable` 中：
```cpp
if (useNdOrientation) {
    const auto [lv0, lv1] = elem.edgeVertices(localEdge);
    sign = (lv0 < lv1) ? 1 : -1;
}
```
当你在后续引入 ND / RT 等 Nedelec (Hcurl) 元素时，请注意 Basix 的边方向可能与 `(lv0 < lv1)` 不一致。在 Lagrange（H1）空间下 `sign` 不改变基函数符号，目前的逻辑可以暂时保留，但当你深度使用高阶 ND 时，可能需要利用 basix 提供的拉回（Piola Transform）矩阵来处理符号，而不是手动打 sign。

### 总结

核心就是：**不要仅在评估（eval）时做数组位置置换，要在拓扑级别（实体）就建立 CCW 到 Basix 的桥梁**。
应用上述 `buildPermutation()` 的实体映射自动生成逻辑后，你的二阶（甚至三阶、四阶）张量积和单纯形单元的问题都将彻底解决，且没有任何冗余判断。
# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
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

针对提供的代码，FE（有限元）层的设计中确实存在一些典型的过度设计（向后兼容遗留）、冗余代码、状态耦合以及对混合网格支持不足的反模式。

为了实现**更加简洁、高效、接口统一且支持混合几何网格**，以下是 FE 层代码的步骤化重构与性能优化方案：

### 第一步：打破“单一几何类型”的假设，原生支持混合网格 (FE Space)

**反模式识别**：在 `fe_space.cpp` 的 `buildDofTable()` 中，直接粗暴地假定了全局网格只有一种几何类型，遇到混合网格直接抛出异常：`"FESpace mixed volume geometries are not supported"`。这不仅限制了混合网格（如四面体+六面体网格），还导致 DOF 偏移计算过于死板。

**重构方案**：
1. 取消对 `meshGeom` 的全局统一检测。
2. 动态计算所有网格实体（Vertex, Edge, Face, Cell）需要的自由度偏移。不再假设所有单元具有相同的 `DofLayout`。
3. 在构建 DOF 表时，根据具体每个 Element 的 `Geometry`，从 `FECollection` 中取出对应的 `ReferenceElement` 和 `DofLayout` 进行局部到全局的映射。

```cpp
// 重构后的 FE Space DOF 分配核心逻辑（概念代码）：
Index offset = 0;
// 1. 遍历所有顶点分配 DOF
for(Index v=0; v < mesh_->numCornerVertices(); ++v) {
    vertexDofOffset_[v] = offset;
    offset += fec_->order() > 0 ? 1 : 0; // H1顶点DOF数
}
// 2. 遍历所有边分配 DOF
for(Index e=0; e < mesh_->numEdges(); ++e) {
    edgeDofOffset_[e] = offset;
    offset += fec_->order() - 1; // H1边内部DOF数
}
// 同理处理 Face 和 Volume，随后在 getElementDofs 中根据单元几何类型组合这些偏移。
```
**优势**：彻底解除混合网格限制，使算法对单元几何完全无感，统一了 DOF 分配接口。

---

### 第二步：消除派生类泛滥，实现 `FiniteElement` 类的泛型化数据驱动

**反模式识别**：在 `finite_element.cpp` 和 `h1.cpp` 中，为每一种几何体硬编码了具体的类（`H1SegmentShape`, `H1TriangleShape`, `H1CubeShape` 等）。这导致了大量重复的模板化代码（Boilerplate），所有的实现都只是在透传给 `GeometryMapping`。

**重构方案**：
完全删除 `H1XXXShape` 系列子类。只保留一个最终类 `H1FiniteElement`，通过构造函数传入 `Geometry` 和 `Order`，用数据驱动替代多态。

```cpp
class H1FiniteElement final : public FiniteElement {
public:
    H1FiniteElement(Geometry geom, int order) : geom_(geom), order_(order) {}
    
    void evalShape(const Vector3& xi, Matrix& shape) const override {
        GeometryMapping::evalShape(geom_, order_, xi, shape); // 直接路由
    }
    
    void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override {
        GeometryMapping::evalDerivatives(geom_, order_, xi, derivatives);
    }
    
    std::vector<int> faceDofs(int faceIdx) const override {
        return buildH1FaceDofs(geom_, order_, faceIdx); // 提取为通用函数
    }
private:
    Geometry geom_;
    int order_;
};
```
**优势**：去除了所有啰嗦的子类和虚函数表冗余，提升了指令缓存（I-Cache）命中率，同时添加新的单元（如 Prism/Pyramid）无需新增任何类。

---

### 第三步：纯化 `GridFunction`，根除 `thread_local` 依赖

**反模式识别**：`grid_function.cpp` 中使用了 `thread_local Matrix t_shapeBuf;` 作为求值缓存。这是一种强烈的反模式：
1. 它引入了隐式的状态。
2. 阻止了同线程内的并发递归（Reentrancy）。
3. GPU移植或基于 Task 的多线程调度（如 TBB）时会导致未定义行为或内存泄漏。

**重构方案**：
利用库中已经定义好的 `MaxNodesPerElement` 常量，直接在栈上分配固定大小的连续内存，完全抛弃 `thread_local` 和动态分配的 Eigen 矩阵。

```cpp
Vector3 GridFunction::gradient(Index elem, const Vector3& xi, const Matrix3& invJacobianTranspose) const {
    const ReferenceElement* ref = fes_->elementRefElement(elem);
    const int nd = ref->numDofs();
    const int vdim = fes_->vdim();
    
    // 完全零堆分配，零线程锁的栈内存
    Eigen::Matrix<Real, MaxNodesPerElement, 3> derivBuf;
    std::array<Index, MaxNodesPerElement * MaxVectorDim> dofsBuf;
    
    ref->basis().evalDerivatives(xi, derivBuf); // 直接写到栈上
    fes_->getElementDofs(elem, std::span{dofsBuf.data(), static_cast<size_t>(nd * vdim)});
    
    Vector3 gRef = Vector3::Zero();
    for (int i = 0; i < nd; ++i) {
        for(int d=0; d<3; ++d) {
            gRef[d] += derivBuf(i, d) * values_[dofsBuf[i * vdim]]; 
        }
    }
    return invJacobianTranspose * gRef;
}
```
**优势**：`GridFunction::eval` 变为了100%纯函数（Pure Function），彻底解放了上层的并发评估限制。

---

### 第四步：提取 `FacetElementTransform` 的拓扑映射查找表 (LUT)

**反模式识别**：在 `facet_element_transform.cpp` 的 `mapToVolumeElement` 函数中，使用了一个巨大的、超过 100 行的 `switch(volGeom) + switch(localFaceIdx)` 瀑布流硬编码逻辑，既难以维护又容易出错。

**重构方案**：
将面到体的参数坐标映射转化为**数据驱动的矩阵乘法**或查表法。面上的局部坐标 $(u, v)$ 可以通过线性映射转换到体坐标 $(x, y, z)$：
$$\mathbf{X}_{vol} = \mathbf{A}_{face} \cdot \mathbf{X}_{face} + \mathbf{b}_{face}$$

将所有的 `A` 矩阵和 `b` 向量预先静态存储在 `geom::` 工具中。
```cpp
bool FacetElementTransform::mapToVolumeElement(const Vector3& bdrXi, Vector3& volXi) const {
    if (adjElemIdx_ == InvalidIndex || localFaceIdx_ < 0) return false;
    
    // 获取 LUT 中面到体的仿射变换矩阵和偏移向量
    const auto& affine = geom::getFaceToVolumeAffineMap(
         mesh_->element(adjElemIdx_).geometry(), localFaceIdx_);
         
    volXi = affine.A * bdrXi + affine.b; // 单指令解决所有几何的映射
    return true;
}
```
**优势**：大幅缩减代码量，根除循环复杂度，同时映射逻辑可以针对所有支持的面-体组合通用。
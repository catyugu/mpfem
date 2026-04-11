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

针对MPFEM有限元（FE）层的代码，我们可以发现多个经典的面向对象（OOP）过度设计、动态内存滥用、以及硬编码带来的冗长问题。由于不需要向后兼容，我们可以大刀阔斧地进行重构。

以下是针对FE层（`fe/`目录）的步骤化重构与性能优化方案，目标是：**零堆分配（Zero Heap Allocation）、扁平化设计、算法化生成基函数**。

---

### 第一步：消除 `ElementTransform` 的动态内存滥用与继承冗余
**反模式与问题：**
`ElementTransform` 内部的 `jacobian_`, `invJacobian_` 使用了 `Matrix` (即 `Eigen::MatrixX`，动态大小)，在 3x3 小矩阵上极大地拖慢了性能，并产生内存碎片。

**重构策略：**。
* 使用最大固定尺寸 `Eigen::Matrix<Real, 3, 3>` 配合 `Eigen::Map` 或 `block` 视图，彻底消除堆分配。

```cpp
// 重构后的 fe/element_transform.hpp
namespace mpfem {

class alignas(64) ElementTransform {
public:
    ElementTransform() = default;

    // 绑定时直接确定是否为边界，记录局部面索引
    void bindElement(Geometry geom, int geomOrder, Index attr, Index elemId, 
                     std::span<const Vector3> nodes, int localFaceIdx = -1) {
        geometry_ = geom;
        geomOrder_ = geomOrder;
        attribute_ = attr;
        elementId_ = elemId;
        dim_ = geom::dim(geom);
        localFaceIdx_ = localFaceIdx; // -1 表示体单元
        numNodes_ = static_cast<int>(nodes.size());
        std::copy(nodes.begin(), nodes.end(), nodesBuf_.begin());
    }

    void setIntegrationPoint(const Vector3& xi);
    
    // 获取法向量
    Vector3 normal() const {
        if (dim_ == 2) {
            return jacobian_.col(0).cross(jacobian_.col(1)).normalized();
        }
        return Vector3::Zero(); // 1D暂略
    }

    // 暴露固定大小矩阵，外部通过 dim() 读取有效列
    const Matrix3& jacobian() const { return jacobian_; }
    const Matrix3& invJacobianT() const { return invJacobianT_; }
    Real weight() const { return weight_; }

private:
    // 数据紧凑排列
    Geometry geometry_ = Geometry::Invalid;
    int geomOrder_ = 1;
    int dim_ = 0;
    int numNodes_ = 0;
    int localFaceIdx_ = -1;
    Index attribute_ = InvalidIndex;
    Index elementId_ = InvalidIndex;

    Real detJ_ = 0.0;
    Real weight_ = 0.0;

    // 使用固定大小的3x3矩阵，彻底干掉 Eigen::MatrixX
    Matrix3 jacobian_ = Matrix3::Zero();
    Matrix3 invJacobian_ = Matrix3::Zero();
    Matrix3 invJacobianT_ = Matrix3::Zero();
    
    std::array<Vector3, MaxNodesPerElement> nodesBuf_;
};

}
```

### 第二步：利用张量积消除 `GeometryMapping` 的千行硬编码
**反模式与问题：**
在 `geometry_mapping.cpp` 中，六面体（Cube）和四边形（Square）的 1 阶和 2 阶形函数与导数是通过手写硬编码实现的，多达上千行。这极其冗长，一旦出错很难排查，且无法支持 3 阶以上单元。

**重构策略：**
* 抛弃硬编码，改用 **1D 张量积 (Tensor Product)** 动态生成 Square 和 Cube 的形函数。
* 对三角形/四面体使用通用的**重心坐标 (Barycentric Coordinates)** 算法。

```cpp
// 重构 geometry_mapping.cpp 的核心逻辑：算法化生成
void GeometryMapping::evalShape(Geometry geom, int order, const Vector3& xi, ShapeMatrix& shape) {
    if (geom == Geometry::Cube) {
        // 通过 1D Segment 的基函数组合出 3D Cube
        ShapeMatrix shape1D_X, shape1D_Y, shape1D_Z;
        evalShape(Geometry::Segment, order, Vector3(xi.x(),0,0), shape1D_X);
        evalShape(Geometry::Segment, order, Vector3(xi.y(),0,0), shape1D_Y);
        evalShape(Geometry::Segment, order, Vector3(xi.z(),0,0), shape1D_Z);
        
        int n1d = order + 1;
        int idx = 0;
        // 算法化遍历生成 3D 节点值，消除几百行硬编码
        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    shape(idx++, 0) = shape1D_X(i,0) * shape1D_Y(j,0) * shape1D_Z(k,0);
                }
            }
        }
        return;
    }
    // 同理处理导数 (DerivMatrix) 的张量积求导律（乘积法则）
    // ...
}
```

### 第三步：`FECollection` 使用 O(1) 静态数组替换 `unordered_map`
**反模式与问题：**
`FECollection` 内部使用 `std::unordered_map<Geometry, std::unique_ptr<ReferenceElement>>` 进行存储。`Geometry` 是一个枚举，其值是小整数。`unordered_map` 会带来不必要的哈希开销、堆内存分配和缓存不友好。

**重构策略：**
使用基于枚举大小的静态 `std::array` 完美实现零开销映射。

```cpp
// fe_collection.hpp
class FECollection {
public:
    // ...
    const ReferenceElement* get(Geometry geom) const {
        return elements_[static_cast<size_t>(geom)].get();
    }
private:
    int order_ = 1;
    Type type_ = Type::H1;
    // 假设 Geometry 种类小于 16，直接用 array
    std::array<std::unique_ptr<ReferenceElement>, 16> elements_; 
};
```

### 第四步：重构 `FESpace` DOF分配，消除维度 `if-else`
**反模式与问题：**
`buildDofTable()` 里充斥着 `if (meshDim == 3)`，并且为 Vertex、Edge、Face、Cell 分别写了四套相似的循环逻辑，极其冗长且难以维护。

**重构策略：**
引入统一的**拓扑实体维度索引 (Entity Dimension, 0~3)**，通过两层循环无缝兼容各种维度。

```cpp
void FESpace::buildDofTable() {
    // 统一用数组记录各类实体需要的 DOF 数量
    // numEntities[0] = numVertices, [1] = numEdges, [2] = numFaces, [3] = numCells
    std::array<int, 4> numEntities = { mesh_->numCornerVertices(), mesh_->numEdges(), mesh_->numFaces(), mesh_->numElements() };
    std::array<std::vector<int>, 4> entityDofs;
    for(int d=0; d<4; ++d) entityDofs[d].resize(numEntities[d], 0);

    // 扫描所有单元，统一填入需要的 DOF
    for (Index e = 0; e < mesh_->numElements(); ++e) {
        const auto layout = fec_->get(mesh_->element(e).geometry())->basis().dofLayout();
        // 获取单元拥有的各种维度的实体索引
        auto vIds = mesh_->getElementVertices(e);
        auto eIds = mesh_->getElementEdges(e);
        auto fIds = mesh_->getElementFaces(e);
        
        for(Index id : vIds) entityDofs[0][id] = std::max(entityDofs[0][id], layout.numVertexDofs);
        for(Index id : eIds) entityDofs[1][id] = std::max(entityDofs[1][id], layout.numEdgeDofs);
        if(mesh_->dim() == 3) {
            for(Index id : fIds) entityDofs[2][id] = std::max(entityDofs[2][id], layout.numFaceDofs);
            entityDofs[3][e] = std::max(entityDofs[3][e], layout.numVolumeDofs);
        } else {
            entityDofs[3][e] = std::max(entityDofs[3][e], layout.numFaceDofs); // 2D cell DOFs
        }
    }

    // 统一进行前缀和累加偏移
    std::array<std::vector<Index>, 4> entityOffsets;
    Index offset = 0;
    for(int d = 0; d < 4; ++d) {
        entityOffsets[d].resize(numEntities[d], 0);
        for(size_t i = 0; i < entityDofs[d].size(); ++i) {
            entityOffsets[d][i] = offset;
            offset += entityDofs[d][i];
        }
    }
    
    scalarNumDofs_ = offset;
    numDofs_ = scalarNumDofs_ * vdim_;
    // 接下来将局部映射写入 elemDofs_，同样用统一的函数数组处理。
}
```

### 第五步：`GridFunction::eval` 彻底静态化与并行安全
**反模式与问题：**
当前的 `GridFunction::eval` 和 `gradient` 会每次动态查询 `getElementDofs` 并拷贝 `values_`，不仅多走了一层间接寻址，在后处理密集调用时会导致性能瓶颈。

**重构策略：**
由于有限元矩阵计算和求值发生在 `ReferenceElement`（高频命中点），我们利用C++ `std::span` 和模板消除临时堆分配，改为就地乘加计算。

```cpp
// fe/grid_function.cpp
Real GridFunction::eval(Index elem, const Vector3& xi) const {
    const ReferenceElement* ref = fes_->elementRefElement(elem);
    const int nd = ref->numDofs();
    const int vdim = fes_->vdim();

    // 复用 ReferenceElement 提供好的、或现算的形函数
    ShapeMatrix shapeBuf;
    ref->basis().evalShape(xi, shapeBuf); // 如果xi是积分点，直接从 ref 提取缓存

    // 避免使用 std::vector/array 拷贝，直接通过指针间接寻址求值
    const Index* elemDofPtr = fes_->getElementDofPointer(elem); // 新增的高效接口
    
    Real val = 0.0;
    // 极速点积展开
    for (int i = 0; i < nd; ++i) {
        val += shapeBuf(i, 0) * values_[elemDofPtr[i] * vdim];
    }
    return val;
}
```

### 总结
通过以上五个维度的重构，去除了 `fe` 层**全部**非必要的运行时堆分配（`Eigen::MatrixX` 到 `Matrix3`、`unordered_map` 到 `std::array`），删减了千行规模的形函数硬编码分支，消除了过度设计的 `FacetElementTransform` 类。内存占用将会大幅减少，尤其是积分装配阶段的 L1 Cache 命中率和整体性能将得到质的飞跃。
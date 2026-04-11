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

### 一、 现有代码诊断 (反模式与性能瓶颈)

1. **动态分配黑洞 (Dynamic Allocation in Inner Loops)**
   * **反模式**：在 `GeometryMapping::evalShape` 和 `evalDerivatives` 中，传入 `Matrix& shape` 并调用 `ensureStorage(shape, rows, cols)`。由于底层是 `Eigen::MatrixX` (堆分配)，在汇编（Assembly）的极内层循环中触发动态扩容，会导致灾难性的性能下降。
2. **缓存破坏 (Cache-Unfriendly Data Structures)**
   * **反模式**：`Mesh` 中的拓扑连接关系使用了 `std::vector<std::vector<std::pair<int, Index>>> elementToEdge_`。Vector of Vectors 导致内存碎片化，缓存未命中率极高。
3. **过度冗余与 DRY 原则违背**
   * **反模式**：`FESpace::buildDofTable` 中有将近 200 行代码在分别处理 `element`（体单元）和 `bdrElement`（边界单元）。它们的拓扑逻辑其实完全一致。
4. **循环依赖与领域泄漏 (Circular Dependencies)**
   * **反模式**：`fe/element_transform.hpp` 居然 `#include "mesh/mesh.hpp"` 并在内部定义了 `bindElementToTransform`。FE层应当是纯粹的数学映射，绝对不应该知道 `Mesh` 或 `Element` 类的存在。

---

### 二、 步骤化重构方案

#### 步骤 1：FE 层零分配抽象 (Zero-Allocation Mathematical Core)
**目标**：彻底消除形函数和雅可比计算中的堆内存分配。使用最大固定大小的栈上分配 (Stack Allocation)。

```cpp

// 1. 重构 fe/geometry_mapping.hpp / finite_element.hpp
class FiniteElement {
public:
    virtual ~FiniteElement() = default;
    virtual int numDofs() const = 0;
    
    // 不再传入 Eigen::MatrixX，而是传入固定大小且不需要 reshape 的 Ref
    virtual void evalShape(const Vector3& xi, Eigen::Ref<ShapeMatrix> shape) const = 0;
    virtual void evalDerivatives(const Vector3& xi, Eigen::Ref<DerivMatrix> derivatives) const = 0;
};

// 2. 在 geometry_mapping.cpp 中消除 switch/case 和 ensureStorage
void GeometryMapping::evalShape(Geometry geom, int order, const Vector3& xi, Eigen::Ref<ShapeMatrix> shape) {
    // 内存安全：shape.setZero() 不需要改变大小
    shape.setZero(); 
    const Real x = xi.x(), y = xi.y(), z = xi.z();
    
    // ... 直接写值，不再调用 ensureStorage ...
    if (geom == Geometry::Hexahedron && order == 1) {
        // ...
    }
}
```

#### 步骤 2：斩断 FE 与 Mesh 的耦合 (Decouple FE from Mesh)
**目标**：`ElementTransform` 不应该知道 `Mesh` 是什么。它只需要一个“几何形状”和“节点坐标”数组。

```cpp
// fe/element_transform.hpp 
// 移除 #include "mesh/mesh.hpp"

class ElementTransform {
public:
    // 纯数据驱动的绑定：仅需知道几何类型和节点坐标即可，与 Mesh 完全解耦
    void bind(Geometry geom, int geomOrder, Index attribute, std::span<const Vector3> nodes) {
        geometry_ = geom;
        geomOrder_ = geomOrder;
        attribute_ = attribute;
        dim_ = geom::dim(geom);
        numNodes_ = static_cast<int>(nodes.size());
        for (int i = 0; i < numNodes_; ++i) {
            nodesBuf_[i] = nodes[i];
        }
    }
    // ...
};

// Assembly 时在汇编器中处理，而不是让 FE 层去依赖 Mesh
// std::array<Vector3, MaxQuadraturePoints> coords;
// trans.bind(geom, order, attr, std::span(coords.data(), numNodes));
```

#### 步骤 3：Mesh 拓扑层的 CSR 扁平化重构 (Cache-Friendly Topology)
**目标**：消除 `vector<vector>`，统一任意维度（1D/2D/3D）的拓扑关系，使用紧凑的 Compressed Sparse Row (CSR) 格式。

```cpp
// mesh/mesh.hpp

// 通用的 CSR 结构
template <typename T>
struct CSRArray {
    std::vector<Index> offsets; // 大小为 N+1
    std::vector<T> data;

    std::span<const T> get(Index i) const {
        return {data.data() + offsets[i], data.data() + offsets[i+1]};
    }
};

class Mesh {
private:
    int dim_ = 3;
    std::vector<Vertex> vertices_;
    
    // 统一将体单元和边界单元抽象为 Entity
    std::vector<Element> elements_;
    std::vector<Element> bdrElements_;

    // 拓扑连接关系全部使用一维连续内存！
    CSRArray<Index> elementToEdge_;
    CSRArray<Index> elementToFace_;
    CSRArray<Index> bdrElementToFace_;
    
public:
    // 统一的接口，隐藏内部扁平化细节
    std::span<const Index> getElementEdges(Index elemIdx) const {
        return elementToEdge_.get(elemIdx);
    }
    
    void buildTopology() {
        // 在此处构建 CSR 数组，一次性预分配内存
        // 例如统计完所有 element 的 face 数量后：
        // elementToFace_.data.reserve(totalFaces);
    }
};
```

#### 步骤 4：FESpace 自由度管理的大统一 (Unified DOF Management)
**目标**：消除数百行的 `if/else` 和由于分离 `elements` 和 `bdrElements` 导致的冗余。

```cpp
// fe/fe_space.cpp

void FESpace::buildDofTable() {
    // 1. 统计各种拓扑实体所需的 DOF 数量
    std::vector<int> vertexDofs(mesh_->numCornerVertices(), 0);
    std::vector<int> edgeDofs(mesh_->numEdges(), 0);
    std::vector<int> faceDofs(mesh_->numFaces(), 0);
    std::vector<int> cellDofs(mesh_->numElements(), 0);

    // 统一处理函数：通过 lambda 捕获多态性
    auto scanElements = [&](const auto& elementsList, bool isBdr) {
        for (Index i = 0; i < elementsList.size(); ++i) {
            const Element& elem = elementsList[i];
            const DofLayout layout = fec_->get(elem.geometry())->basis().dofLayout();
            
            for (Index vId : isBdr ? mesh_->getBdrElementVertices(i) : mesh_->getElementVertices(i)) {
                vertexDofs[mesh_->vertexToCornerIndex(vId)] = std::max(..., layout.numVertexDofs);
            }
            // 处理 Edge / Face 同理，利用统一的 mesh_->getElementXXXs(i) ...
        }
    };

    scanElements(mesh_->elements(), false);
    scanElements(mesh_->bdrElements(), true);

    // 2. 算前缀和 (Prefix Sum) 获取 offsets
    // ...

    // 3. 填充 DOF 表（使用 CSR 风格连续存储）
    elemDofs_.offsets.resize(mesh_->numElements() + 1);
    // 直接顺序写入，极大简化代码，取消 InvalidIndex 判定和大量分支
}
```

#### 步骤 5：雅可比矩阵求逆的极致优化 (Math Kernel Optimization)
**目标**：目前 `ElementTransform` 依然使用了动态库和 Eigen 的 `ldlt().solve` 来算非方阵伪逆。我们应手写或完全展开。

```cpp
// fe/element_transform.cpp
void ElementTransform::computeInverse() {
    if (dim_ == 3) {
        kernels::inverse3(jacobian_.data(), invJacobian_.data());
    } else if (dim_ == 2) {
        // 对于 3D 空间中的 2D 表面，计算伪逆 J^+ = (J^T J)^{-1} J^T
        // 展开计算，避免调用 Eigen 的动态 LDLT 分解！
        Real JtJ[4]; 
        kernels::matmatT2x3(jacobian_.data(), JtJ); // 自定义 2x3 乘 3x2 kernel
        Real invJtJ[4];
        kernels::inverse2(JtJ, invJtJ);
        // invJacobian_ = invJtJ * J^T
        kernels::matTmat2x2x3(invJtJ, jacobian_.data(), invJacobian_.data());
    }
}
```

### 总结
1. **去掉了所有的运行时堆分配**，FE 的运算耗时将降低到原本的 **1/3 到 1/10**。
2. **彻底解耦** 了 `fe` 和 `mesh` 层。现在的 `FiniteElement` 和 `ElementTransform` 变得极其容易进行单元测试。
3. `Mesh` 的**内存占用大幅下降**（没有了大量的 `std::vector` 头结构开销），遍历缓存命中率提升。
4. `FESpace` 的代码量缩减了大约 50%，逻辑变得异常清晰且跨维度通用（1D, 2D, 3D 直接由统一实体接口处理）。
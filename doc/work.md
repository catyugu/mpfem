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

针对你指出的核心痛点（**语义混淆、严重的H1偏见、自由度分配依赖几何节点而非拓扑**），这是一个经典的早期FEM框架走向通用多物理场FEM框架时必经的重构阵痛。

当前的架构强行将“定义几何形状的函数”和“定义物理场的试探/检验函数”绑定在一起，导致不仅无法支持L2（间断Galerkin）、ND（电磁边缘元）、RT（流体力学面元），甚至连协调的次参数/超参数（Sub/Super-parametric）网格处理也是通过 Hack 的方式实现的。

以下是彻底消除这些反模式、走向现代化通用FEM架构的**步骤化重构方案**（完全放弃向后兼容）：

---

### 第一步：解耦几何插值与物理场基函数 (Decouple Geometry vs Physics)

**问题代码：** `ElementTransform` 内部包含了一个 `std::unique_ptr<FiniteElement> geoBasis_;`。这导致几何映射复用了物理场的类，迫使 `FiniteElement` 不得不具备几何坐标相关的属性。

**重构行动：**
剥离几何与物理。引入轻量级的 `GeometryEvaluator` 或 `ShapeMapping` 纯粹用于计算等参映射 $\mathbf{x}(\xi)$ 和雅可比矩阵，彻底把 `FiniteElement` 从 `ElementTransform` 中踢出去。

```cpp
// 新增：专门处理几何映射的静态类或轻量级计算器（完全摒弃 FiniteElement）
class GeometryMapping {
public:
    // 仅针对 Geometry (Triangle, Hexahedron 等) 和几何阶数计算形函数
    static void evalShape(Geometry geom, int order, const Vector3& xi, VectorX& shape);
    static void evalDerivatives(Geometry geom, int order, const Vector3& xi, MatrixX& dshape);
};

// 重构 ElementTransform
class ElementTransform {
    // 移除 std::unique_ptr<FiniteElement> geoBasis_;
    // 移除 geometryOrder 与 fieldOrder 混淆的概念
public:
    void computeJacobianAtIP() {
        // 直接使用专用的几何形函数，与物理场基函数彻底解绑
        GeometryMapping::evalDerivatives(geometry_, geomOrder_, ip_.getXi(), geoShapeDerivatives_);
        // 计算 J = X * dPhi ...
    }
};
```

---

### 第二步：建立真正的网格拓扑体系 (Formalize Mesh Topology)

**问题代码：** 目前的 `Mesh` 仅管理了 Vertex 和 Element，所谓的 `buildTopology()` 只是建立了 Element 到 Face 的映射，**缺失了全局唯一 Edge 和全局唯一 Face 的实体编号体系**。

**重构行动：**
有限元自由度是附着在**拓扑实体**（0D顶点、1D边、2D面、3D体）上的，而不是几何节点上。必须让网格生成完整的拓扑关联。

```cpp
class Mesh {
public:
    // 获取完整的拓扑数量
    Index numVertices() const;
    Index numEdges() const;     // 新增
    Index numFaces() const;     // 新增
    Index numElements() const;
    
    // 单元需要能返回其包含的拓扑实体在全局中的 ID
    std::vector<Index> getElementVertices(Index elemIdx) const;
    std::vector<Index> getElementEdges(Index elemIdx) const;    // 新增
    std::vector<Index> getElementFaces(Index elemIdx) const;    // 新增
};
```

---

### 第三步：重设计 FiniteElement 接口 (Eliminate H1 Bias in FE)

**问题代码：** 当前的 `FiniteElement` 包含 `dofsPerVertex()`, `dofCoords()` 等带有严重 H1 (拉格朗日元) 偏见的接口。这导致根本无法定义例如“内部有4个自由度、面上无自由度的L2元”。

**重构行动：**
引入 `DofLayout`（自由度拓扑布局）签名，彻底消除与几何坐标的直接关联。

```cpp
// 定义自由度在不同维度拓扑实体上的分布签名
struct DofLayout {
    int numVertexDofs = 0; // 附着在0维顶点的自由度数
    int numEdgeDofs = 0;   // 附着在1维边的自由度数
    int numFaceDofs = 0;   // 附着在2维面的自由度数
    int numVolumeDofs = 0; // 附着在3维体内部的自由度数
};

class FiniteElement {
public:
    virtual ~FiniteElement() = default;
    virtual BasisType basisType() const = 0; // H1, L2, ND, RT
    
    // 核心重构：返回该单元的拓扑自由度签名
    virtual DofLayout getDofLayout() const = 0;
    
    // 移除 dofCoords()！ 物理基函数不需要知道自己在物理空间中的绝对坐标！
    
    // 计算参考单元上的基函数值与梯度
    virtual void evalShape(const Vector3& xi, Matrix& shape) const = 0;
    virtual void evalDerivatives(const Vector3& xi, Matrix& dshape) const = 0;
};
```

*举例：*
* `H1_1阶`：`DofLayout{1, 0, 0, 0}` (只在顶点有DoF)
* `L2_1阶`：`DofLayout{0, 0, 0, 4}` (只在体内部有4个DoF)
* `ND_1阶`(Nedelec)：`DofLayout{0, 1, 0, 0}` (只在边上有1个DoF)

---

### 第四步：重写 FESpace 自由度分配 (Topology-based DoF Allocation)

**问题代码：** `FESpace::buildDofTable` 中直接将 DoF 映射给 `elem.vertex(j)`，这是万恶之源。

**重构行动：**
依据各维度的全局拓扑数量，连续分配自由度。无论什么元，分配逻辑将高度统一、极度简洁且无漏洞。

```cpp
void FESpace::buildDofTable() {
    const ReferenceElement* refElem = fec_->get(Geometry::Tetrahedron); // 假设统一网格
    DofLayout layout = refElem->basis().getDofLayout();
    
    // 1. 计算各个拓扑维度的全局 DoF 偏移量
    Index v_offset = 0;
    Index e_offset = v_offset + mesh_->numVertices() * layout.numVertexDofs;
    Index f_offset = e_offset + mesh_->numEdges()    * layout.numEdgeDofs;
    Index c_offset = f_offset + mesh_->numFaces()    * layout.numFaceDofs;
    
    numDofs_ = c_offset + mesh_->numElements() * layout.numVolumeDofs;
    
    // 2. 为每个单元分配局部到全局的 DoF 映射
    elemDofs_.resize(mesh_->numElements() * refElem->numDofs());
    
    for (Index e = 0; e < mesh_->numElements(); ++e) {
        Index localDofIdx = 0;
        const Index base = e * refElem->numDofs();
        
        // 映射顶点自由度
        for (Index vId : mesh_->getElementVertices(e)) {
            for (int k = 0; k < layout.numVertexDofs; ++k)
                elemDofs_[base + localDofIdx++] = v_offset + vId * layout.numVertexDofs + k;
        }
        // 映射边自由度
        for (Index eId : mesh_->getElementEdges(e)) {
            for (int k = 0; k < layout.numEdgeDofs; ++k)
                elemDofs_[base + localDofIdx++] = e_offset + eId * layout.numEdgeDofs + k;
        }
        // 映射面自由度
        for (Index fId : mesh_->getElementFaces(e)) {
            for (int k = 0; k < layout.numFaceDofs; ++k)
                elemDofs_[base + localDofIdx++] = f_offset + fId * layout.numFaceDofs + k;
        }
        // 映射体（内部）自由度
        for (int k = 0; k < layout.numVolumeDofs; ++k) {
            elemDofs_[base + localDofIdx++] = c_offset + e * layout.numVolumeDofs + k;
        }
    }
}
```
**收益：** 这一段不超过 30 行的代码，能够直接完美兼容 H1, L2, Nedelec, Raviart-Thomas，彻底消灭了原来的 `elem.vertex(j)` 补丁和所谓的 "Subparametric/Superparametric" 逻辑冗余。

---

### 第五步：改造 Dirichlet 边界条件 (Refactor Dirichlet BCs)

**问题代码：** `applyDirichletBC` 强依赖 `dofCoords`，试图找到几何节点的物理坐标去算解析解，完全把 FEM 降维成了 FDM（有限差分）。

**重构行动：**
对于本质边界条件，不应该依赖几何坐标，而应利用 **边界拓扑的面积分点投影 (L2 Projection on Boundary)** 或 **节点自由度泛函 (Node Functionals)**。最简洁的改法是，让 FESpace 根据拓扑直接返回边界面上激活的全局 DoF 列表，并通过边界积分点评估 `VariableNode`。

```cpp
void applyDirichletBC(...) {
    // 遍历所有施加了BC的边界拓扑面
    for (Index b = 0; b < mesh.numBdrElements(); ++b) {
        if (!isTargetBoundary(mesh.bdrElement(b).attribute())) continue;
        
        // 获取该拓扑面上的所有全局自由度（得益于第四步的重构，可以直接获取，包括边/面上的DoF）
        std::vector<Index> bdrDofs = fes.getBdrElementDofs(b);
        
        // 针对 H1，计算边界积分点上的投影或者直接使用边界顶点做 nodal interpolation。
        // 这里不再需要 dofCoords，而是：通过 ElementTransform 遍历边界单元上的局部求值点
        // 将解固定。
        for (size_t i = 0; i < bdrDofs.size(); ++i) {
            Index gDof = bdrDofs[i];
            if (eliminated[gDof]) continue;
            
            // 依据自由度的泛函定义(Node Functional)在局部计算对应的值，赋给全局
            Real val = evaluateBoundaryValue(coef, trans, local_dof_index); 
            dofVals[gDof] = val;
            eliminated[gDof] = true;
        }
    }
}
```

### 总结

这五步重构彻底**砍断了“自由度”与“几何点”之间的孽缘**。
1. **GeometryMapping** 负责 $\hat{x} \to x$ (仅仅关心几何与Jacobian)。
2. **Mesh Topology** 负责点、边、面、体的连接关系。
3. **FiniteElement/DofLayout** 负责说明方程自由度的数学分布。
4. **FESpace** 就像一个矩阵，将 Topology 和 DofLayout 正交相乘，完成全局映射。

重构后不仅代码会大幅缩减（因为去掉了为填补概念漏洞而写的海量 `if (order_ == ...)`），性能也会因为数据结构变成了紧凑的拓扑数组而提升。
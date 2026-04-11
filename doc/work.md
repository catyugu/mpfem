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

针对您代码库中的设计反模式、H1偏见、接口混乱以及几何与物理场语义不分等紧要问题，我为您制定了**彻底抛弃向后兼容的步骤化重构方案**。

目标是：**语义清晰、零开销抽象、现代 C++ 范式 (C++20/23)、消除类型和指针混用**。

---

### 步骤一：接口层“大扫除”（解决 Span/指针/引用 混用的困惑）
**反模式识别**：当前接口混合使用了 `Real* x`、`Matrix&`、`std::vector`、`std::span`。这对用户（甚至内部开发者）意味着隐式的内存所有权和生命周期危机。
**重构方案**：全面使用现代 C++ 的轻量级视图，并拥抱 RVO (返回值优化) 和 `Eigen::Ref`。

**1. 消除裸指针和输出参数（出参）**
将所有需要填充的“出参”直接作为返回值（如果是小对象）或 `Eigen::Ref`（如果是大矩阵）。
```cpp
// ❌ 重构前：混用、语义不清
virtual void transform(const Vector3& xi, Real* x);
void evalShape(const Vector3& xi, Matrix& shape) const;

// ✅ 重构后：利用 RVO 返回小对象，利用 Eigen::Ref 传递矩阵映射
virtual Vector3 transform(const Vector3& xi) const;
virtual void evalShape(const Vector3& xi, Eigen::Ref<Eigen::MatrixXd> shape) const;
```

**2. 统一内存视图**
凡是连续内存的输入，一律使用 `std::span<const T>`；不再传递 `std::vector` 或使用裸指针配大小。

---

### 步骤二：彻底分离“几何插值基函数”与“物理场基函数”
**反模式识别**：现在的 `FiniteElement` 既被用作网格几何的映射（在 `ElementTransform::geoBasis_` 中），又被用作物理方程的形函数。这在理论上和代码上都是灾难（因为几何变换通常永远是连续的 H1，而物理场可能是 L2, H(div) 或 H(curl)）。

**重构方案**：将其拆分为两个完全正交的概念：`GeometryShape` 和 `FieldBasis`。

**1. 几何映射器 (GeometryShape)**
纯粹为了计算雅可比矩阵和物理坐标，**永远只返回标量和标量梯度**。
```cpp
class GeometryShape {
public:
    virtual ~GeometryShape() = default;
    virtual int numNodes() const = 0;
    // 强制输出确定的类型
    virtual void evalValues(const Vector3& xi, std::span<Real> values) const = 0;
    virtual void evalDerivatives(const Vector3& xi, Eigen::Ref<Eigen::MatrixX3d> derivs) const = 0;
};
```
在 `ElementTransform` 中，我们**只**使用 `GeometryShape`，不再关心物理场的自由度。

**2. 物理场基函数 (FEBasis)**
专门用于物理方程，支持不同类型的张量输出。
```cpp
class FEBasis {
public:
    virtual ~FEBasis() = default;
    virtual int numDofs() const = 0;
    virtual BasisType type() const = 0; // H1, L2, ND(Hcurl), RT(Hdiv)
    
    // 返回泛型的张量数组，H1返回标量，ND返回向量
    virtual void evalBasis(const Vector3& xi, std::span<Tensor> basisValues) const = 0;
    // ...
};
```

---

### 步骤三：破除严重的“H1 偏见” (Refactoring FESpace)
**反模式识别**：`FESpace::buildDofTable` 中存在着 `elemDofs_[base + j] = elem.vertex(j)` 这种极度硬编码的逻辑。它直接将 DOF 绑定到网格的 Vertex 上，使得加入 L2（无节点，仅体自由度）或 Nedelec（仅边自由度）变得不可能。

**重构方案**：基于网格的**拓扑实体 (Topological Entities)** 来分配自由度，而不是基于顶点。

```cpp
void FESpace::buildDofTable() {
    Index currentGlobalDof = 0;

    // 1. 为所有网格顶点分配 DOF (针对 H1)
    vertexDofOffset_.resize(mesh_->numVertices());
    for (Index v = 0; v < mesh_->numVertices(); ++v) {
        vertexDofOffset_[v] = currentGlobalDof;
        currentGlobalDof += fec_->dofsPerVertex();
    }

    // 2. 为所有边分配 DOF (针对 Nedelec 或 高阶 H1)
    edgeDofOffset_.resize(mesh_->numEdges());
    for (Index e = 0; e < mesh_->numEdges(); ++e) {
        edgeDofOffset_[e] = currentGlobalDof;
        currentGlobalDof += fec_->dofsPerEdge();
    }

    // 3. 为所有面分配 DOF (针对 RT 或 高阶 H1/ND)
    faceDofOffset_.resize(mesh_->numFaces());
    for (Index f = 0; f < mesh_->numFaces(); ++f) {
        faceDofOffset_[f] = currentGlobalDof;
        currentGlobalDof += fec_->dofsPerFace();
    }

    // 4. 为所有体单元分配 DOF (针对 L2 或 超高阶)
    volumeDofOffset_.resize(mesh_->numElements());
    for (Index el = 0; el < mesh_->numElements(); ++el) {
        volumeDofOffset_[el] = currentGlobalDof;
        currentGlobalDof += fec_->dofsPerVolume();
    }

    numDofs_ = currentGlobalDof * vdim_;
}
```
**效果**：现在 `FESpace` 变得极其通用。如果是 L2 元素，`dofsPerVertex`, `dofsPerEdge` 自动返回 `0`，自由度只会精准落在 Volume 上。

---

### 步骤四：斩断循环依赖与对象冗余
**反模式识别**：`VariableManager` 与 `VariableNode` 在解析(`resolve`)期间形成了互相引用的强绑定；积分器 (`Integrator`) 严重依赖 `ReferenceElement` 这艘“大航母”。

**重构方案**：
**1. 将表达式评估无状态化 (Functional AST)**
去除 `CompiledExpressionNode` 中持有对其他 Node 的原始指针（`resolved_deps_`）。改为在 `EvaluationContext` 中携带一个轻量的上下文或直接通过 ID 去 `Manager` 里查，斩断对象的生命周期耦合。

**2. 积分器接口的依赖倒置 (Dependency Inversion)**
不要让积分器（如 `DiffusionIntegrator`）去拿整个 `ReferenceElement`。积分器只需要两样东西：**物理梯度（或值）** 和 **雅可比权重**。
```cpp
// ❌ 重构前：积分器承担了太多获取梯度的职责
void assembleElementMatrix(const ReferenceElement& ref, ElementTransform& trans, Matrix& elmat) const;

// ✅ 重构后：让核心层去求好基函数的值，喂给积分器。积分器只做一件事：做乘法和累加！
void assembleIntegrationPoint(Real weight, std::span<const Tensor> basisVals, std::span<const Tensor> basisGrads, Eigen::Ref<Eigen::MatrixXd> elmat) const;
```
这样不仅代码大幅减少，由于分支和指针解引用在循环内被消灭，缓存命中率(Cache Locality) 将会有质的飞跃。

### 总结
经过上述重构，您可以：
1. **删除 `FiniteElement` 类的多重职责**，拆解为 `GeometryShape` 和 `FieldBasis`。
2. **重写 `FESpace`**，完全基于网格拓扑维度（Vertex, Edge, Face, Cell）来分发自由度。
3. **接口现代化**，全盘禁止裸指针，消灭 `std::vector` 的按值返回，改为 `std::span` 和 `Eigen::Ref`。
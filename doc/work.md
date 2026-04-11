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

这是一个非常典型且关键的有限元（FEM）软件架构演进问题。当前代码正处于从“只能做简单的标量H1等参元”向“通用多物理场、多单元类型有限元框架”过渡的阵痛期。

你指出的问题非常准确：**几何与物理的混淆、Basis 抽象的缺失、拓扑硬编码以及接口风格的混乱**，是制约代码扩展为支持 Nedelec（电磁场）、Raviart-Thomas（流体/混合列式）或 L2（间断伽辽金）等基底的根本障碍。

以下是针对这些反模式的**步骤化重构方案**和现代 C++ 代码实现。

---

### 第一步：诊断与反模式清除

1. **反模式：几何映射与物理基底混用 (God ShapeFunction)**
   * **现状**：`ElementTransform`（用于计算坐标映射和雅可比）直接调用了 `ShapeFunction::create`。
   * **重构**：几何映射应当使用独立的 `GeometricFactors` 或专用的 `IsoparametricBasis`。物理场基底（如 Nedelec）决不能用来做几何插值。
2. **反模式：硬编码拓扑的 `ReferenceElement` (Topological Coupling)**
   * **现状**：`ReferenceElement::faceDofs` 里通过 `if (order_ >= 2)` 和先角点、后边的硬编码逻辑来分配自由度。这导致一旦引入没有角点 DoF 的 Nedelec 边元，代码直接崩溃。
   * **重构**：将 DoF 的布局分布（DofLayout）职责交还给基底（Basis）自身。
3. **反模式：不支持向量基底 (Scalar Assumption)**
   * **现状**：`evalValues` 输出 `std::span<Real>`，`evalGrads` 输出 `std::span<Vector3>`。如果是 ND 或 RT 基底，基函数本身就是向量，散度是标量，旋度是向量，现有签名无法兼容。
   * **重构**：使用统一的秩（Rank）和分量（vdim）抽象，结合 `Matrix&` 消除 `span/指针/引用` 混用，提高内存互操作性。

---

### 第二步：核心接口重构设计

我们将原有的 `ShapeFunction` 废弃，引入真正的 **`FiniteElement` (或叫 `Basis`)** 接口。

#### 1. 定义单元与自由度的规范 (finite_element.hpp)

```cpp
#ifndef MPFEM_FINITE_ELEMENT_HPP
#define MPFEM_FINITE_ELEMENT_HPP

#include "mesh/geometry.hpp"
#include <Eigen/Dense>
#include <vector>

namespace mpfem {

// 定义基底的类型，区分连续性与张量特性
enum class BasisType {
    H1, // 标量连续元 (Lagrange)
    L2, // 标量间断元 (DG)
    ND, // 矢量边元 (Nedelec - H(curl))
    RT  // 矢量面元 (Raviart-Thomas - H(div))
};

// 抽象基函数接口（隔离了具体单元的内部实现）
class FiniteElement {
public:
    virtual ~FiniteElement() = default;

    virtual BasisType basisType() const = 0;
    virtual Geometry geometry() const = 0;
    virtual int order() const = 0;
    virtual int numDofs() const = 0;

    // 基函数的值的维度：标量元(H1, L2)=1，矢量元(ND, RT)=空间维度(通常是2或3)
    virtual int vdim() const = 0;

    // --- 核心解耦：让基底自己报告其拓扑分布 ---
    virtual int dofsPerVertex() const = 0;
    virtual int dofsPerEdge() const = 0;
    virtual int dofsPerFace() const = 0;
    virtual int dofsPerVolume() const = 0;

    // --- 统一评估接口 ---
    // shape: [numDofs x vdim]
    // 使用 Eigen::Ref 无缝对接后续的汇编计算，消灭 span 和 原始指针
    virtual void evalShape(const Eigen::Vector3d& xi, 
                           Matrix& shape) const = 0;

    // derivatives: 
    // 对于 H1: [numDofs x 空间维度] (Gradient)
    // 对于 ND: [numDofs x 空间维度] (Curl)
    // 对于 RT: [numDofs x 1] (Divergence)
    virtual void evalDerivatives(const Vector& xi, 
                                 Matrix& derivatives) const = 0;

    // 获取局部DoF在参考单元内的坐标（用于插值或Dirichlet边界条件）
    virtual void getDofCoords(std::vector<Vector>& coords) const = 0;
};

} // namespace mpfem

#endif
```

#### 2. 实现具体的基底 (H1 与 几何基底分离)

创建一个专门处理拉格朗日（H1）多项式的具体实现。这不仅解决了泛型问题，还可以作为几何映射的 `GeometricBasis`。

```cpp
#include "finite_element.hpp"

namespace mpfem {

class H1_TriangleElement : public FiniteElement {
public:
    explicit H1_TriangleElement(int order) : order_(order) {}

    BasisType basisType() const override { return BasisType::H1; }
    Geometry geometry() const override { return Geometry::Triangle; }
    int order() const override { return order_; }
    int numDofs() const override { return (order_ + 1) * (order_ + 2) / 2; }
    int vdim() const override { return 1; }

    // 拓扑映射：一阶只有顶点，二阶有顶点和边
    int dofsPerVertex() const override { return 1; }
    int dofsPerEdge() const override { return order_ - 1; }
    int dofsPerFace() const override { return (order_ > 2) ? ((order_ - 1)*(order_ - 2)/2) : 0; }
    int dofsPerVolume() const override { return 0; }

    void evalShape(const Vector& xi, 
                   Matrix& shape) const override {
        // shape size is [numDofs x 1]
        Real x = xi.x(), y = xi.y();
        if (order_ == 1) {
            shape(0, 0) = 1.0 - x - y;
            shape(1, 0) = x;
            shape(2, 0) = y;
        } else if (order_ == 2) {
            shape(0, 0) = (1.0 - x - y) * (1.0 - 2.0 * x - 2.0 * y);
            // ... (其他二阶形函数)
        }
    }

    void evalDerivatives(const Vector& xi, 
                         Matrix& derivatives) const override {
        // derivatives size is [numDofs x 2] (或3，取决于空间维)
        if (order_ == 1) {
            derivatives.row(0) << -1.0, -1.0, 0.0;
            derivatives.row(1) << 1.0,  0.0, 0.0;
            derivatives.row(2) << 0.0,  1.0, 0.0;
        }
    }

    void getDofCoords(std::vector<Vector>& coords) const override {
        // 返回插值点...
    }

private:
    int order_;
};

} // namespace mpfem
```

#### 3. 将缓存器职责限制在 ReferenceElement 中

原来的 `ReferenceElement` 是一个“上帝对象”，承担了 DoF 映射的职责。现在我们将它退化为一个**不可变的缓存器 (Evaluator Cache)**，并彻底拥抱 Eigen 矩阵。

```cpp
#ifndef MPFEM_REFERENCE_ELEMENT_HPP
#define MPFEM_REFERENCE_ELEMENT_HPP

#include "finite_element.hpp"
#include "quadrature.hpp"

namespace mpfem {

/**
 * @brief ReferenceElement 仅负责在积分点上对基函数的值和导数进行缓存计算。
 * 不再涉及拓扑映射，完全适配任意类型的基底。
 */
class ReferenceElement {
public:
    ReferenceElement(std::unique_ptr<FiniteElement> fe, int quadratureOrder)
        : fe_(std::move(fe)) {
        quadrature_ = quadrature::get(fe_->geometry(), quadratureOrder);
        precompute();
    }

    const FiniteElement& basis() const { return *fe_; }
    const QuadratureRule& quadrature() const { return quadrature_; }
    int numQuadraturePoints() const { return quadrature_.size(); }

    // 使用 Eigen::Map 或 const Eigen::MatrixXd& 返回预计算的数据
    // shapeValues[q] 返回 [numDofs x vdim] 的矩阵
    const Matrix& shapeValuesAtQuad(int q) const {
        return cachedShapeValues_[q];
    }
    
    // shapeDerivatives[q] 返回 [numDofs x derivativeDim] 的矩阵
    const Matrix& shapeDerivativesAtQuad(int q) const {
        return cachedDerivatives_[q];
    }

private:
    void precompute() {
        int nq = quadrature_.size();
        cachedShapeValues_.resize(nq);
        cachedDerivatives_.resize(nq);

        for (int q = 0; q < nq; ++q) {
            const auto& xi = quadrature_[q].getXi();
            
            // 依据基底属性动态分配内存缓存 (对于 H1, L2, ND 等一视同仁)
            cachedShapeValues_[q].resize(fe_->numDofs(), fe_->vdim());
            
            // 假设统一空间导数矩阵维度为 3
            cachedDerivatives_[q].resize(fe_->numDofs(), 3); 

            fe_->evalShape(xi, cachedShapeValues_[q]);
            fe_->evalDerivatives(xi, cachedDerivatives_[q]);
        }
    }

    std::unique_ptr<FiniteElement> fe_;
    QuadratureRule quadrature_;
    std::vector<Matrix> cachedShapeValues_;
    std::vector<Matrix> cachedDerivatives_;
};

}
#endif
```

#### 4. 重构 FE_Space：通用拓扑分发

得益于 `FiniteElement` 能够自己报告 `dofsPerVertex()`, `dofsPerEdge()`，`FESpace` 中的 `buildDofTable` 现在可以非常优雅，**直接支持混合网格、任意阶数和任意单元类型**。

```cpp
void FESpace::buildDofTable() {
    // 1. 初始化每个拓扑实体的偏置(Offsets)
    int numVertices = mesh_->numVertices();
    int numEdges = mesh_->numEdges();
    int numFaces = mesh_->numFaces();
    int numVolumes = mesh_->numElements();

    // 假设所有单元同构（如果非同构，需要累加）
    const FiniteElement& fe = *fec_->get(mesh_->element(0).geometry());
    
    int vDofs = fe.dofsPerVertex();
    int eDofs = fe.dofsPerEdge();
    int fDofs = fe.dofsPerFace();
    int volDofs = fe.dofsPerVolume();

    // 计算全局 DoF 数量
    numDofs_ = (numVertices * vDofs) + 
               (numEdges * eDofs) + 
               (numFaces * fDofs) + 
               (numVolumes * volDofs);

    // ... 按照偏移量依次为每个单元分配全局 ID，彻底消灭 `if(order >= 2)` 的硬编码
}
```

### 重构效果总结

1. **接口统一与高效**：全面弃用 `Real*`, `std::span` 杂搭，统一采用 `Eigen::MatrixXd&`。这一举措使得基函数的求值与组装器（Assembler）中计算局部刚度矩阵时的线性代数操作无缝衔接，直接利用 SIMD 指令提速。
2. **面向对象的 Basis 概念**：通过引入 `FiniteElement` 基类并赋予 `vdim` 概念，系统现在天生支持标量（H1/L2）与向量（ND/RT）形函数，后续引入诸如 `evalCurl` 等只需要派生相应的 Element 类。
3. **消除硬编码与循环依赖**：`ReferenceElement` 被剥离了与几何拓扑挂钩的职能，仅充当 `FiniteElement` 的**多态缓存包装器**，杜绝了代码中随处可见的 `switch(order)` 或 `if(subparametric)` 散弹弹片，逻辑变得极为紧凑。
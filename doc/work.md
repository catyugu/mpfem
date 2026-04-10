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

针对当前代码库中存在的深度耦合、接口设计僵化和向后兼容包袱，我们需要进行一次彻底的、破坏性的（无须向后兼容）重构。当前的核心痛点在于：**标量H1单元（Lagrange）的假设被硬编码到了系统的各个角落**（如 `ShapeFunction` 返回 `Real` 和 `Vector3`，`EvaluationContext` 硬编码逆雅可比转置等）。

为了支持 Nedelec（$H(curl)$）和 Raviart-Thomas（$H(div)$）单元，必须引入 **Piola 变换**，并严格分离**几何映射（Geometry Mapping）**与**物理场基函数（FE Basis）**。

以下是分步骤的重构指南，每一步均保持可编译和可验证：

### 第一步：解耦 `EvaluationContext` 中的几何硬编码

**目标**：移除 `EvaluationContext` 中针对 H1 单元硬编码的 `invJacobianTransposes`，改为传递统一的 `ElementTransform` 集合，使表达式系统可以根据需要获取雅可比矩阵、行列式或 Piola 变换因子。

**重构操作**：
修改 `expr/evaluation_context.hpp`：
```cpp
#ifndef MPFEM_EXPR_EVALUATION_CONTEXT_HPP
#define MPFEM_EXPR_EVALUATION_CONTEXT_HPP

#include "core/tensor_shape.hpp"
#include "core/tensor.hpp"
#include "core/types.hpp"
#include <span>

namespace mpfem {

    class ElementTransform; // 前置声明

    struct EvaluationContext {
        Real time = Real(0);
        int domainId = -1;
        Index elementId = InvalidIndex;
        std::span<const Vector3> physicalPoints;
        std::span<const Vector3> referencePoints;
        
        // 重构：移除 std::span<const Matrix> invJacobianTransposes;
        // 替换为统一的变换接口引用，使得算子可以随时获取 J, detJ, invJ 等
        std::span<ElementTransform* const> transforms;
    };

} // namespace mpfem
#endif
```

**验证**：修改 `assembly/integrators.cpp` 中的 `makeSinglePointContext` 和 `dirichlet_bc.hpp`，将传入 `invJTs` 改为传入 `&trans` 的 span。编译通过即可。

---

### 第二步：分离几何形函数与有限元基函数，引入 `MapType`

**目标**：现在的 `ShapeFunction` 既被用作几何坐标变换，又被用作物理场求解。我们需要将物理场基函数抽象为 `FiniteElement`，并引入 `MapType`（映射类型）来处理 Nedelec 和 RT 单元的 Piola 变换。

**重构操作**：
新建 `fe/finite_element.hpp`：
```cpp
#ifndef MPFEM_FINITE_ELEMENT_HPP
#define MPFEM_FINITE_ELEMENT_HPP

#include "core/types.hpp"
#include "fe/element_transform.hpp"

namespace mpfem {

    // 物理量从参考单元到物理单元的映射类型 (Piola transforms)
    enum class MapType {
        VALUE,     // H1 (Lagrange): u = u_ref
        H_CURL,    // Nedelec (Edge): u = J^{-T} * u_ref
        H_DIV,     // Raviart-Thomas (Face): u = (J / detJ) * u_ref
        L2_INTEGRAL// L2: u = u_ref / detJ
    };

    class FiniteElement {
    public:
        virtual ~FiniteElement() = default;

        virtual int numDofs() const = 0;
        virtual int dim() const = 0;
        virtual MapType mapType() const = 0;

        // 获取参考单元上的基函数值（具体子类实现对应维度的张量）
        virtual void calcShape(const Real* xi, Real* shape) const { MPFEM_THROW(NotImplementedException, ""); }
        virtual void calcVShape(const Real* xi, MatrixX& shape) const { MPFEM_THROW(NotImplementedException, ""); }
        
        // 获取物理单元上的基函数值（内置 Piola 变换）
        virtual void calcPhysShape(ElementTransform& trans, Real* shape) const;
        virtual void calcPhysVShape(ElementTransform& trans, MatrixX& shape) const;
    };

} // namespace mpfem
#endif
```
**说明**：将现有的 `H1TriangleShape` 等重构为继承自 `FiniteElement`（针对场变量）和 `GeometryShape`（纯几何坐标变换，仅保留 VALUE 映射）。

---

### 第三步：重构 `ReferenceElement` 抹除标量场假设

**目标**：当前的 `ReferenceElement` 深度耦合了 H1 单元的假设，其内部预先分配了 `std::vector<Real> shapeValues_` 和 `std::vector<Vector3> shapeGradients_`。对于 RT/Nedelec 单元，基函数本身就是向量，不支持标量梯度。

**重构操作**：
修改 `fe/reference_element.hpp`，使用多态的缓存或者让 `FiniteElement` 来决定缓存类型：
```cpp
class ReferenceElement {
public:
    ReferenceElement(Geometry geom, std::unique_ptr<FiniteElement> fe);

    const FiniteElement* fe() const { return fe_.get(); }
    const QuadratureRule& quadrature() const { return quadrature_; }

    // 移除硬编码的 shapeValue 和 shapeGradient，改为由 Assembler 动态向 FE 索取
    // 或者提供统一的基于 Tensor 的预计算接口
    
private:
    Geometry geometry_;
    std::unique_ptr<FiniteElement> fe_;
    QuadratureRule quadrature_;
    // 缓存数据应当被重构为一个通用的 Tensor 数组，或者推迟到 Assembler 中基于 ThreadBuffer 运算
};
```
在这一步，你将 `DiffusionIntegrator` 中直接调用 `ref.shapeGradientsAtQuad(q)` 的逻辑，修改为通过 `FiniteElement::calcPhysDShape(trans, dshape)` 实时或在积分点循环前统一获取物理域梯度。这彻底解除了积分器对 ReferenceElement 缓存结构的依赖。

### 总结

通过上述四步重构：
1. **表达式系统** (`EvaluationContext`) 不再被 H1 绑架，支持了各种复杂映射。
2. **基函数** (`FiniteElement`) 与 **几何** (`ShapeFunction`) 彻底分离，引入 Piola 变换机制。
3. **缓存机制** (`ReferenceElement`) 退回本职工作，不再强加标量场和梯度的假设，为 Nedelec (Curl) 和 RT (Div) 提供空间。
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

这是一个非常典型的有限元（FE）代码库重构场景。代码中反映出了早期 C++（指针、裸数组）向现代 C++（span、Eigen、自定义 Tensor）过渡过程中的“历史包袱”，导致接口碎片化；同时，“标量基函数”与“几何映射”的强绑定也是一个经典的有限元设计反模式，严重阻碍了后续扩展（比如引入 Nedelec 边单元或 RT 面单元）。

为了达到**简洁、高效、接口统一（抛弃向后兼容）**的目标，我们需要进行分步重构。每一步都保证是自包含且可编译的。

---

### 第一步：统一坐标与数学类型（消除裸指针与类型碎片）

**问题识别**：
在 `fe` 层中，积分点参考坐标 `xi` 混用了 `const Real*` 和 `IntegrationPoint`；自由度坐标返回了低效的 `std::vector<std::vector<Real>>`；`ElementTransform` 中存在大量互相调用的重载接口。

**重构操作**：
1. 统一所有空间坐标点为 Eigen 的 `Vector3`（对于 1D/2D，未使用的分量默认为 0）。
2. 删除所有 `const Real* xi` 的冗余接口。
3. `dofCoords` 统一返回 `std::vector<Vector3>`。

**代码修改**：
修改 `fe/shape_function.hpp` 和 `fe/element_transform.hpp`：
```cpp
// 1. 修改 ShapeFunction 接口
class ShapeFunction {
public:
    virtual ~ShapeFunction() = default;
    virtual Geometry geometry() const = 0;
    virtual int order() const = 0;
    virtual int numDofs() const = 0;
    virtual int dim() const = 0;

    // 彻底废弃 const Real* xi 和 Real* values，全部改为 Vector3 和 span
    virtual void evalValues(const Vector3& xi, std::span<Real> values) const = 0;
    virtual void evalGrads(const Vector3& xi, std::span<Vector3> grads) const = 0;

    // 返回值统一为 Vector3 数组
    virtual std::vector<Vector3> dofCoords() const = 0; 

    static std::unique_ptr<ShapeFunction> create(Geometry geom, int order);
};

// 2. 修改 ElementTransform 接口，移除所有冗余重载
class ElementTransform {
public:
    // ...
    void setIntegrationPoint(const IntegrationPoint& ip);
    void setIntegrationPoint(const Vector3& xi); // 替代 const Real*

    Vector3 transform(const Vector3& xi) const;  // 直接返回，消除输出参数
    Vector3 transformGradient(const Vector3& refGrad) const; 

    // 内部缓存改为使用 Vector3
    std::array<Real, MaxNodesPerElement> shapeValuesBuf_;
    std::array<Vector3, MaxNodesPerElement> shapeGradsBuf_;
};
```
*实现细节更新*：在 `shape_function.cpp` 中，提取 `xi` 的值统一改为 `const Real x = xi.x(); const Real y = xi.y();`。

**验证方式**：编译整个 `fe` 文件夹，确保所有基于下标的指针访问 `xi[0], xi[1]` 均被替换为 Eigen 向量的访问。

---

### 第二步：解耦几何插值与物理场有限元（Finite Element vs Geometry Mapping）

**问题识别**：
目前 `ElementTransform`（几何映射）和 `ReferenceElement`（物理场）都直接调用 `ShapeFunction::create`。这隐含了一个致命假设：**所有的物理场基函数都可以用于几何变换，且都是标量的（H1）**。如果未来引入矢量场（如 Nedelec 单元），一个基函数在某点求值将是一个 `Vector3`，当前的 `ShapeFunction` 接口（`evalValues` 返回 `Real`）将彻底崩溃。

**重构操作**：
1. 将物理场的概念抽象为 `FiniteElement`。
2. 将纯几何标量插值独立为 `GeoBasisFunction`。
3. `ElementTransform` **仅**依赖 `GeoBasisFunction`，而 `ReferenceElement` 依赖 `FiniteElement`。

**代码修改**：
```cpp
// fe/finite_element.hpp (新文件)
namespace mpfem {

// 抽象的物理场有限元基类
class FiniteElement {
public:
    virtual ~FiniteElement() = default;
    virtual int numDofs() const = 0;
    virtual int dim() const = 0;
    
    // 返回类型：H1为标量，ND/RT为矢量
    enum class RangeType { Scalar, Vector };
    virtual RangeType rangeType() const = 0;

    // 统一的接口：无论是标量还是矢量场，都写入预分配的 Eigen::Matrix / Vector 中
    // 对于标量单元：shapeMatrix 是 numDofs x 1
    // 对于矢量单元：shapeMatrix 是 numDofs x dim
    virtual void calcShape(const Vector3& xi, MatrixX& shapeMatrix) const = 0;
    
    // 计算参考坐标下的旋度或散度或梯度，取决于单元类型
    virtual void calcDShape(const Vector3& xi, MatrixX& dShapeMatrix) const = 0;
};

// 保留原有的 ShapeFunction，但更名为 GeoBasisFunction，专门服务于 ElementTransform
class GeoBasisFunction {
    // ... 保持原来的 evalValues 和 evalGrads，专门用于坐标变换
};
}
```
**验证方式**：修改 `ElementTransform` 使其内部只实例化 `GeoBasisFunction`，修改 `ReferenceElement` 使其持有 `FiniteElement`。编译通过即可验证几何与物理已成功切断耦合。

---

### 第三步：重构 GridFunction 和 Assembler 消除反模式和冗长（Remove Thread-Local Anti-Pattern）

**问题识别**：
`GridFunction::eval` 中使用了 `thread_local std::vector`（为了避免堆分配），这是一个严重的设计反模式：当出现嵌套求值或重入时会导致数据竞争和覆写。同时它强行假设标量场并手动乘以 `vdim`。Assembler 中的 `ThreadBuffer` 定义了极大的栈数组，冗长且浪费内存。

**重构操作**：
废弃 `GridFunction` 中的线程局部缓存。直接利用 Eigen 的栈分配特性或将其分配责任转移给 `EvaluationContext`。

**代码修改**：
修改 `fe/grid_function.cpp`：
```cpp
// 彻底移除 thread_local 缓存！
// 利用 Eigen 的栈上动态大小特性（如最多支持64个DOF），避免堆分配
using MaxDofVector = Eigen::Matrix<Real, Eigen::Dynamic, 1, 0, 64, 1>;

VectorX GridFunction::eval(Index elem, const Vector3& xi) const {
    if (!fes_) return VectorX::Zero(vdim());

    const ReferenceElement* ref = fes_->elementRefElement(elem);
    const FiniteElement* fe = ref->finiteElement();
    
    int nd = fe->numDofs();
    int vdim = fes_->vdim();
    
    // 栈上分配（如果超过 64 则会自动退化为堆分配，但多数单元不会超过）
    MaxDofVector shapeVals(nd); 
    fe->calcShape(xi, shapeVals); // 重构后的接口

    std::vector<Index> dofs(nd * vdim);
    fes_->getElementDofs(elem, dofs);

    VectorX result = VectorX::Zero(vdim);
    for (int i = 0; i < nd; ++i) {
        for(int c = 0; c < vdim; ++c) {
            result(c) += shapeVals(i) * values_[dofs[i * vdim + c]];
        }
    }
    return result; // 现代 C++ 中 RVO 会优化掉拷贝
}
```
**验证方式**：使用多线程组装矩阵或并行计算误差时，结果不再发生偶发性错误（消除了隐蔽的 thread_local 污染）。代码也从几十行缩减为十几行。

---

### 第四步：打破循环依赖（Header Cleanup）

**问题识别**：
`assembler.hpp` 包含了 `integrator.hpp`, `sparse_matrix.hpp` 等。`fe_space.hpp` 包含了 `mesh.hpp` 和 `fe_collection.hpp`。这导致了更改任何一个底层类都会触发几乎整个工程的重编译。

**重构操作**：
使用严格的前向声明，在头文件中只包含绝对必要的类型。

**代码修改**：
重构 `fe/fe_space.hpp`：
```cpp
#ifndef MPFEM_FE_SPACE_HPP
#define MPFEM_FE_SPACE_HPP

#include "core/types.hpp"
#include <span>
#include <memory>
#include <vector>

// 1. 全部改为前向声明
namespace mpfem {
class Mesh;
class FECollection;
class ReferenceElement;
enum class Geometry : std::uint8_t;

class FESpace {
public:
    FESpace();
    FESpace(const Mesh* mesh, std::unique_ptr<FECollection> fec, int vdim = 1);
    ~FESpace(); // 2. 析构函数必须在 cpp 中实现，因为 unique_ptr 需要知晓 FECollection 的完整大小

    // 移除内联实现中对 mesh->xxx 的调用，全部移到 .cpp 文件
    const Mesh* mesh() const;
    void getElementDofs(Index elemIdx, std::span<Index> dofs) const;
    
    // ...
private:
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    int vdim_ = 1;
    // ...
};
}
#endif
```

将原本写在 `fe_space.hpp` 底部的 `inline void FESpace::getElementDofs` 等所有包含具体逻辑的代码，全部迁移到 `fe_space.cpp` 中。

**验证方式**：
触碰 `mesh.hpp` 后，执行构建。你会发现由于打破了宏观的包含链，只有少数几个 `.cpp` 文件被重新编译，构建速度会有数量级的提升。

---

### 总结
通过以上四步：
1. **接口统一**：消除了 `Real*`, `array`, `Tensor` 之间混乱的传参，统一收敛到 `Eigen::Vector3`。
2. **架构高效解耦**：将几何变换与物理插值剥离，为支持后续高阶电磁场单元（ND/RT）扫清了障碍。
3. **消除反模式**：杀掉了极度危险且冗长的 `thread_local` 缓存和过度设计的 `ThreadBuffer`。
4. **编译加速**：清理了纠缠不清的头文件。
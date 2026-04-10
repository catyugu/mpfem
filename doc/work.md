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

通过分析提供的源码，当前架构中存在非常典型的 **God Class (上帝类)** 和 **职责过载 (Overloaded Responsibilities)** 反模式。

最核心的设计缺陷在于：`ShapeFunction` 类被**双重使用**：
1. 它被 `ElementTransform` 用于计算**网格几何的坐标映射**（Isoparametric mapping，即参考单元到物理单元的映射）。
2. 它又被 `ReferenceElement` 用于计算**物理场变量的基函数**（如温度、位移的插值）。

这导致了**标量基函数（当前假定）与矢量基函数（如将来的电磁场边单元、流体面单元）无法分离**，且内部包含巨大的硬编码 `switch-case`，严重违反了开闭原则 (OCP)。

以下是消除这些设计反模式、解耦模块并提高性能的步骤化重构方案（无需向后兼容）：

---

### 步骤 1：将 `ShapeFunction` 职责分离，定义全新的 `FiniteElement` 体系
**目的**：为物理场基函数建立独立的接口体系，区分标量场与矢量场，取代旧的 `ShapeFunction`。

**重构动作**：
创建一套专门服务于物理场的基类接口：

```cpp
// 1. 所有物理场基函数的顶层接口
class FiniteElement {
public:
    virtual ~FiniteElement() = default;
    virtual Geometry geometry() const = 0;
    virtual int order() const = 0;
    virtual int numDofs() const = 0;
    
    // 决定该单元在参考坐标系下的维度类型（标量=1，矢量=3等）
    virtual int vdim() const = 0; 
    
    // 返回参考坐标系下的自由度节点坐标
    virtual const std::vector<std::vector<Real>>& dofCoords() const = 0;
};

// 2. 标量基函数接口 (例如连续的 H1 Lagrange 单元, 不连续的 L2 单元)
class ScalarFiniteElement : public FiniteElement {
public:
    int vdim() const final { return 1; }
    virtual void evalShape(const Real* xi, Real* values) const = 0;
    virtual void evalGradients(const Real* xi, Vector3* grads) const = 0;
};

// 3. 矢量基函数接口 (例如 Nedelec 边单元, Raviart-Thomas 面单元)
class VectorFiniteElement : public FiniteElement {
public:
    int vdim() const final { return 3; }
    virtual void evalVectorShape(const Real* xi, Vector3* values) const = 0;
    virtual void evalCurls(const Real* xi, Vector3* curls) const = 0;
    virtual void evalDivs(const Real* xi, Real* divs) const = 0;
};
```
*【验证】*：此时代码可以编译。新的接口明确了标量与矢量的区别，这正是未来引入其他类型单元（如电磁场的边单元）的关键。

---

### 步骤 2：使用依赖注入 (DI) 解耦 `ReferenceElement`
**反模式**：目前 `ReferenceElement` 在其构造函数中通过 `ShapeFunction::create(...)` 硬编码实例化形状函数，隐藏了依赖，产生了深度耦合。

**重构动作**：
将 `ReferenceElement` 改为一个“纯数据容器”，只负责持有预计算的数据和积分点。它的构造不再负责创建基函数对象。

```cpp
class ReferenceElement {
public:
    // 通过依赖注入传入 FiniteElement，不拥有其生命周期
    ReferenceElement(const FiniteElement* fe, QuadratureRule quad)
        : fe_(fe), quadrature_(std::move(quad)) 
    {
        numDofs_ = fe_->numDofs();
        precomputeShapeValues(); // 内部可以调用 fe_->evalShape 等
    }

    const FiniteElement* fe() const { return fe_; }
    // ... 其他方法保持不变
private:
    const FiniteElement* fe_ = nullptr; 
    QuadratureRule quadrature_;
    std::vector<Real> shapeValues_;
    std::vector<Vector3> shapeGradients_;
};
```
*【验证】*：`ReferenceElement` 彻底与具体的几何多项式实现解绑。

---

### 步骤 3：重构 `FECollection` 作为单元工厂
**反模式**：以前使用全局的 `switch-case` 工厂（`ShapeFunction::create`）。
**优化**：将其转变为多态的 `FECollection`（如 `H1_FECollection`, `ND_FECollection`），由集合统一管理内存并生成 `ReferenceElement`。

**重构动作**：
```cpp
class H1_FECollection : public FECollection {
public:
    explicit H1_FECollection(int order) : order_(order) {
        // 在这里显式注册不同几何的标量 Lagrange 单元
        elements_[Geometry::Triangle] = std::make_unique<H1TriangleElement>(order);
        elements_[Geometry::Tetrahedron] = std::make_unique<H1TetrahedronElement>(order);
        // ... 
        
        // 预组装 ReferenceElement
        for (auto& [geom, fe] : elements_) {
            auto quad = quadrature::get(geom, 2 * order);
            refElements_[geom] = std::make_unique<ReferenceElement>(fe.get(), quad);
        }
    }
    
    const ReferenceElement* get(Geometry geom) const override {
        return refElements_.at(geom).get();
    }
private:
    int order_;
    std::unordered_map<Geometry, std::unique_ptr<FiniteElement>> elements_;
    std::unordered_map<Geometry, std::unique_ptr<ReferenceElement>> refElements_;
};
```
*【验证】*：将旧的 `H1TriangleShape` 类重命名为 `H1TriangleElement`，实现 `ScalarFiniteElement` 接口，并放入此集合中。开闭原则 (OCP) 得到满足，未来新增单元只需新建类，无需修改核心组装代码。

---

### 步骤 4：解耦 `ElementTransform` 并消除多余内存分配
**冗长与性能反模式**：`ElementTransform` 结构体内目前持有一个 `std::unique_ptr<ShapeFunction> geoShapeFunc_`。在多线程并行装配矩阵时，频繁的动态内存分配会导致极大开销。几何坐标映射只需要最基础的无状态 Lagrange 插值即可。

**重构动作**：
创建一个静态、无状态的 `GeometryInterpolator`，完全弃用原来的 `ShapeFunction` 类。

```cpp
class GeometryInterpolator {
public:
    // 无状态的静态方法：输入局部坐标，输出几何映射梯度
    static void evalGrads(Geometry geom, int order, const Real* xi, Vector3* grads) {
        if (geom == Geometry::Triangle && order == 1) {
            grads[0] = Vector3(-1.0, -1.0, 0.0);
            grads[1] = Vector3(1.0, 0.0, 0.0);
            grads[2] = Vector3(0.0, 1.0, 0.0);
        }
        // ... (原 ShapeFunction 的无状态几何部分转移到此处)
    }
    
    static void evalValues(Geometry geom, int order, const Real* xi, Real* values);
};
```

修改 `ElementTransform`：
```cpp
class ElementTransform {
    // 移除 std::unique_ptr<ShapeFunction> geoShapeFunc_; 彻底去除堆分配

    void computeJacobianAtIP() {
        // 直接调用无状态静态插值器，零内存分配开销
        GeometryInterpolator::evalGrads(geometry_, geomOrder_, &ip_.xi, shapeGradsBuf_.data());
        
        // J = sum_i (x_i * grad_phi_i^T)
        jacobian_.setZero(spaceDim_, dim_);
        for (int i = 0; i < numNodes_; ++i) {
            const auto& grad = shapeGradsBuf_[i];
            for (int d = 0; d < spaceDim_; ++d) {
                for (int k = 0; k < dim_; ++k) {
                    jacobian_(d, k) += nodesBuf_[i][d] * grad[k];
                }
            }
        }
        // ... (计算 Jacobian 逆的代码保持不变)
    }
};
```
*【验证】*：旧的 `ShapeFunction` 相关文件可被完全删除。`ElementTransform` 不再依赖复杂的类层次结构，其仅专注于几何雅可比计算，性能大幅提升。

### 总结
这套重构方案执行完毕后，原本耦合在一起的乱麻将被划分为三条清晰的防线：
1. **`GeometryInterpolator`**: 仅负责几何映射（无状态、极快）。
2. **`FiniteElement` 继承体系**: 专门定义场基函数，天然支持标量/矢量区分（解决引入新单元的根本障碍）。
3. **`FECollection` 与 `ReferenceElement`**: 用作配置与组装，用依赖注入替代了硬编码 switch-case 逻辑。
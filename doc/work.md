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

### 第一步：消除积分热点的 AST 语法树求值 (性能核心优化)

**识别反模式：过度抽象 (Over-abstraction in Hot Path)**
在 `assembly/integrators.cpp` 中，每次计算积分点时（`for (int q = 0; q < nq; ++q)`），都会调用 `evalScalarNode`。而 `evalScalarNode` 底层会调用表达式语法树 (AST) 进行遍历求值。在紧凑的有限元核心循环中进行树遍历和虚函数调用，会带来灾难性的性能（Cache Miss 和分支预测失败）惩罚。

**重构策略：批处理与预计算**
由于 `VariableNode` 已经支持 `evaluateBatch`，我们应将变量求值提取到积分循环**外部**。

```cpp
// 【重构后】以 MassIntegrator 为例
void MassIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                           ElementTransform& trans,
                                           Matrix& elmat) const
{
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    elmat.setZero(nd, nd);

    // 1. 在栈上预分配数组，避免动态内存分配
    std::array<Tensor, MaxQuadraturePoints> coefValues;
    
    // 2. 批量构建该单元所有积分点的 EvaluationContext 并一次性求值！
    // 这彻底消除了内部循环的 AST 解析开销
    EvaluationContext batchCtx = trans.buildBatchContext(ref.quadrature());
    coef_->evaluateBatch(batchCtx, std::span(coefValues.data(), nq));

    // 3. 干净、极速的纯数学积分循环
    for (int q = 0; q < nq; ++q) {
        trans.setIntegrationPoint(ref.integrationPoint(q).getXi());
        const Real w = ref.integrationPoint(q).weight * trans.weight();
        const Real coef = coefValues[q].scalar(); // O(1) 数组读取
        
        const Matrix& phi = ref.shapeValuesAtQuad(q);
        const auto phiVec = phi.col(0);
        elmat.noalias() += w * coef * (phiVec * phiVec.transpose());
    }
}
```

### 第二步：重构 ElementTransform 机制 (性能优化)

**识别反模式：微优化导致的分支预测失败 (Branching for Lazy Evaluation)**
`ElementTransform` 内部使用了 `evalFlags_`（如 `EVAL_JACOBIAN`）来做延迟求值（Lazy Evaluation）。在 FEM 组装中，Jacobian、DetJ 和 Inverse 在 99% 的情况下都是**必定会用到**的。每次调用 `jacobian()` 都去执行 `if (!(evalFlags_ & EVAL_JACOBIAN))` 会打破 CPU 的流水线预测。

**重构策略：积极求值 (Eager Evaluation)**
丢弃 `evalFlags_`，在 `setIntegrationPoint` 时直接计算所有几何映射量，使访问器变为无分支的内联函数。

```cpp
// 【重构后】fe/element_transform.hpp
class ElementTransform {
public:
    void setIntegrationPoint(const Vector3& xi) {
        ipXi_ = xi;
        // 积极求值：舍弃 Lazy Evaluation
        computeJacobian();
        computeInverse();
    }

    // 访问器变成极速的内联返回，没有任何 if 判断
    inline const Matrix& jacobian() const { return jacobian_; }
    inline Real weight() const { return weight_; }
    inline const Matrix& invJacobianT() const { return invJacobianT_; }
    // ...
};
```

### 第三步：拆解 GeometryMapping 上帝类 (设计模式优化)

**识别反模式：上帝类与巨大的 Switch (God Class)**
`fe/geometry_mapping.cpp` 中的 `evalShape` 和 `evalDerivatives` 包含了数百行的 `switch-case`，将一维线段、二维三角形/四边形、三维四面体/六面体的逻辑全塞在一个函数里。严重违反开闭原则 (OCP)。

**重构策略：策略模式 (Strategy Pattern)**
为几何映射引入多态策略（或模板特化），利用工厂/注册表模式解耦。

```cpp
// 1. 定义几何形函数基类
class ShapeFunction {
public:
    virtual ~ShapeFunction() = default;
    virtual void evalShape(int order, const Vector3& xi, Matrix& shape) const = 0;
    virtual void evalDerivatives(int order, const Vector3& xi, Matrix& derivatives) const = 0;
};

// 2. 隔离具体实现
class HexahedronShape : public ShapeFunction {
public:
    void evalShape(int order, const Vector3& xi, Matrix& shape) const override {
        // 仅保留 Cube 的形函数逻辑 ...
    }
};

// 3. GeometryMapping 变为无分支的路由工厂
class GeometryMapping {
    static const ShapeFunction& getShape(Geometry geom) {
        // 内部静态注册表，消除 Switch
        static const std::unordered_map<Geometry, std::unique_ptr<ShapeFunction>> registry = {
            {Geometry::Cube, std::make_unique<HexahedronShape>()},
            {Geometry::Tetrahedron, std::make_unique<TetrahedronShape>()}
        };
        return *registry.at(geom);
    }
public:
    static void evalShape(Geometry geom, int order, const Vector3& xi, Matrix& shape) {
        getShape(geom).evalShape(order, xi, shape);
    }
};
```

### 第四步：清理 Assembler 中的过度设计 (简洁性优化)

**识别反模式：冗杂的线程缓冲 (Over-engineering Thread Buffer)**
在 `assembly/assembler.cpp` 中，作者手动维护了 `ThreadBuffer`，并在循环中尝试复用和 `resize` 动态矩阵 `dynMatrix` 以适配多线程。实际上，Eigen 对于小维度矩阵（如 27x27 的 `MaxVectorDofsPerElement`）在栈上（Stack）分配的速度远超在堆上做尺寸维护。

**重构策略：利用 Eigen 的静态/栈分配**
废弃 `ThreadBuffer` 的动态管理，直接利用 C++ 作用域和 Eigen 宏观静态尺寸。

```cpp
// 【重构后】Assembler 内部循环
#pragma omp parallel
{
    std::vector<SparseMatrix::Triplet> localTriplets;
    localTriplets.reserve(estimatedTriplets / omp_get_num_threads());
    ElementTransform trans;
    
    // 直接在栈上分配最大可能的局部矩阵，自动销毁且线程绝对安全
    Eigen::Matrix<Real, MaxVectorDofsPerElement, MaxVectorDofsPerElement> elmat;

    #pragma omp for schedule(dynamic, 64)
    for (Index e = 0; e < numElements; ++e) {
        // 绑定逻辑...
        elmat.setZero(); // 栈上清零极快
        
        for (auto& integ : domainIntegs_) {
            // integrator 直接写入定长栈矩阵，不再需要传递 dynMatrix 重新 resize
            integ->assembleElementMatrix(*ref, trans, elmat); 
        }

        // 写入 Triplet...
    }
    // 合并 localTriplets ...
}
```

### 第五步：解除 FESpace 与 Mesh 拓扑的循环依赖

**识别反模式：权责不清的拓扑查询**
在 `fe_space.cpp` 的 `buildDofTable` 方法中，`FESpace` 深度干涉了 `Mesh` 的底层结构，通过大量嵌套循环调用 `getElementVertices` / `getElementEdges` / `getElementFaces` 来拼凑自由度映射。

**重构策略：统一的拓扑视图**
在 `Mesh` 中提供一个 `ElementTopology` 结构，将拓扑聚合一次性返回，让 `FESpace` 成为单纯的消费者。

```cpp
// 在 mesh.hpp 中新增
struct ElementTopology {
    std::vector<Index> vertices;
    std::vector<Index> edges;
    std::vector<Index> faces;
};

// 【重构后】fe_space.cpp
void FESpace::buildDofTable() {
    for (Index e = 0; e < mesh_->numElements(); ++e) {
        // 一次性获取统一视图
        const ElementTopology topo = mesh_->getElementTopology(e);
        
        // 直接按顺序映射自由度，省去原来大量 mapVertexDof、mapEdgeDof 的 lambda 闭包逻辑
        for (Index vId : topo.vertices) {
            // 分配顶点自由度...
        }
        for (Index edgeId : topo.edges) {
            // 分配边自由度...
        }
    }
}
```

### 总结
这套重构方案丢弃了向后兼容的包袱：
1. **接口层面**：通过策略模式（GeometryMapping）和统一上下文（EvaluationContext）实现了接口规范化。
2. **性能层面**：利用**预计算（移出 AST）**和**积极求值（移除分支）**打通了核心积分的任督二脉，能带来肉眼可见的速度飞跃。
3. **简洁层面**：利用 Eigen 栈分配代替手动状态机，使得 Assembler 的多线程代码直观易读。
# 大纲

## 前言

* **极其重要**: 你必须首先阅读我们项目的 `README.md`
* 这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
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

## 任务

这是一项非常关键的架构重构。在深入分析您的 `mpfem` 源码后，我发现目前在有限元（FE）、网格（Mesh）和组装（Assembly）层之间存在一些典型的**设计反模式、冗余代码和不合理的依赖关系**。

以下是针对您要求的缺陷分析以及步骤化的重构计划。

### 🚨 识别出的架构问题

1. **依赖倒置与层级破坏 (FESpace 越权与依赖污染)**：
   * **问题**：`src/fe/fe_space.cpp` 严重依赖 `src/mesh/mesh.hpp`。`FE` 层本应是纯粹的参考单元和形函数数学库，完全不应知道全局网格、拓扑或内存分布。这导致了 `FE` -> `Mesh` 的反向依赖，破坏了模块的单向流动。
2. **违反开闭原则 (FECollection 的 Factory 反模式)**：
   * **问题**：当前的 `FECollection` 使用了 `enum class Type { H1, L2, ND, RT }` 和内置的 Switch-Case 工厂方法来创建。如果您未来想添加新的单元类型，就必须修改这个核心类的代码，这是典型的反模式。
3. **职责混淆 (FESpace 管理 vdim)**：
   * **问题**：`FESpace` 构造时接收了 `vdim`（向量维度），并在获取 DOF 时进行 `for (int c = 0; c < vdim_; ++c)` 的硬编码循环。实际上，Nedelec 单元（虽然物理上是 3 维矢量）的自由度是在边上（`vdim` 概念对其并不适用）。让 `FESpace` 管理 `vdim` 导致了泛用性的丧失。
4. **Assembler 中的冗长与分支 (Verbose Array Expansions)**：
   * **问题**：在 `Assembler::assemble()` 中，为了兼容标量积分器和矢量 `vdim`，强行写了 `if (ivdim == 1) { 展开到对角块 } else { 直接加 }`。这会在最热的循环中引入不必要的分支和冗余的复制。

---

### 🛠️ 步骤化重构计划 (强制不向后兼容)

#### **步骤 1：抽象 FECollection，下放 vdim 职责 (解决开闭原则与 vdim 管理)**
**目标**：剥离 `FESpace` 对 `vdim` 的管理，使用面向对象继承替代 Enum。

1. 修改 `FECollection` 为纯虚基类：
   ```cpp
   class FECollection {
   public:
       virtual ~FECollection() = default;
       virtual const ReferenceElement* get(Geometry geom) const = 0;
       virtual std::string name() const = 0;
   };
   ```
2. 创建 `H1_Collection` 继承体系：
   * 只有 H1 需要 `vdim`。添加 `H1_Collection : public FECollection`。
   * 构造函数签名改为：`H1_Collection(int order, int vdim = 1)`。
   * 它在内部直接初始化带有 `vdim` 信息的 `H1FiniteElement`。
3. `H1FiniteElement` 存储 `vdim_`。其形函数矩阵大小直接变为 `[ndof * vdim, vdim]`。

#### **步骤 2：基底注册自由度布局 (DofLayout)**
**目标**：消除 FESpace 中关于几何体、维度、阶数的硬编码猜测。

1. 在 `FiniteElement` 抽象类中引入一个标准化的 `DofLayout` 结构体：
   ```cpp
   struct DofLayout {
       int numVertexDofs = 0;
       int numEdgeDofs = 0;
       int numFaceDofs = 0;
       int numVolumeDofs = 0;
   };
   ```
2. **H1 注册**：在 `H1FiniteElement` 构造时，根据 `vdim` 和 `order` 注册自己的 Layout：
   * 线性 `H1_Collection(1, vdim=3)`：`numVertexDofs = 3`, `numEdgeDofs = 0`。
   * 二阶 `H1_Collection(2, vdim=1)`：`numVertexDofs = 1`, `numEdgeDofs = 1`，如果是四边形则还有 `numFaceDofs = 1`。
3. 将此信息直接暴露给 `ReferenceElement`。

#### **步骤 3：迁移并纯化 FESpace (解决依赖污染)**
**目标**：将 `FESpace` 从 `src/fe/` 移动到 `src/assembly/` 层，变成纯粹的映射器。

1. **移动文件**：将 `fe_space.hpp/cpp` 移到 `assembly` 模块下。这样 `fe` 模块就完全与 `mesh` 解耦，纯粹处理数学。
2. **重写全局自由度分配**：
   * `FESpace` **不再知道什么是 H1 还是 vdim**。
   * 遍历拓扑：根据 `FECollection` 提供的 `DofLayout`，自动在 `Mesh` 的各个拓扑实体（节点、边、面、体）上按需分配全局连续的 DOF 编号。
   * 删除所有类似于 `for(c=0; c<vdim)` 的嵌套逻辑。分配器只需要问：“这条边需要几个 DOF？” Layout 答：“3个”，分配器就直接连续分配 3 个。

#### **步骤 4：集成一阶与二阶 Nedelec (ND) 单元**
**目标**：验证架构是否已真正通用，支持棱边元。

1. 创建 `ND_Collection : public FECollection`，构造函数只接收 `order`。
2. 创建 `NDFiniteElement`：
   * **一阶 Nedelec 布局**：`numVertexDofs = 0`, `numEdgeDofs = 1`。
   * **二阶 Nedelec 布局**：`numVertexDofs = 0`, `numEdgeDofs = 2`, `numFaceDofs = 2` (对四面体)。
3. **引入 Piola 变换**：
   * Node 单元使用标准插值：$u(x) = \hat{u}(\hat{x})$。
   * Nedelec 单元表示矢量场（如电场、磁场），**必须**使用协变 Piola 变换（Covariant Piola Transform）保证旋度连续（$H(curl)$）：
     $$\mathbf{u}(\mathbf{x}) = J^{-T} \hat{\mathbf{u}}(\hat{\mathbf{x}})$$
   * 在 `FiniteElement` 或 `ElementTransform` 接口中，提供一个通用的变换评估函数 `transformBasis(...)`，ND 单元重载此函数并乘上 `trans.invJacobianT()`。

#### **步骤 5：清理 Assembler 的冗长与内存优化**
**目标**：简化装配层，提升热点性能。

1. **去除冗余分支**：由于现在的 `H1_Collection` 已经直接在 `FiniteElement` 层合并了 `vdim`，`Integrator` 返回的直接是 `[totalDof x totalDof]` 的满阵。
   * 删除 `Assembler.cpp` 中 `if (ivdim == 1)` 的展开逻辑，直接使用 `+= buf.dynMatrix`。
2. **预分配机制（Pre-allocation）优化**：
   * `FESpace` 由于彻底变成了 Flat 结构，`fes->getElementDofs` 现在直接以 `span` 返回一维的平铺索引数组。无需在 `Assembler` 内进行二次 `vdim` 乘法。

---

### 🚀 执行建议 (执行此重构的 Checklist)

请遵循您在 `work.md` 中制定的原则，严格**放弃向后兼容**。

1. [ ] 删掉 `fe_collection.hpp/cpp` 中的工厂模式。实现 `H1_Collection`。
2. [ ] 在 `finite_element.hpp` 增加 `DofLayout`。修改 `H1FiniteElement`。
3. [ ] 将 `fe_space.cpp/hpp` 移动到 `assembly` 目录。更新其构建 `DofTable` 的逻辑。
4. [ ] 消除 `Assembler::assemble` 中的分支判断代码。
5. [ ] 编译并运行一阶、二阶稳态测试，确保以上 H1 架构重构未破坏结果。
6. [ ] 提交代码（第一次 Checkpoint）。
7. [ ] 添加 `ND_Collection` 和 `NDFiniteElement`。
8. [ ] 更新 `ElementTransform` 及 `Integrator`，使其支持 `ND` 的 Piola 变换规则。
9. [ ] 编译通过，完成验证。
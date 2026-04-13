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

基于提供的代码和设计宪法，我为您深入分析了 `mpfem` 项目当前存在的设计反模式、冗长代码、向后兼容累赘以及潜在的漏洞，并为您制定了**彻底不向后兼容**的步骤化重构计划。

### 🚨 识别出的设计反模式与缺陷

1. **`ElementTransform` 接口膨胀与职责错位**
   * **反模式**：`ElementTransform` 既负责几何映射（Jacobian计算），又越俎代庖地提供了 `transformGradient` 等特定于 H1 节点元的方法。
   * **后果**：它强行假定所有求导都需要逆转置雅可比矩阵（$J^{-T}$），完全封死了向 Nedelec（需要协变 Piola 变换）或 Raviart-Thomas（需要逆变 Piola 变换）扩展的可能。
2. **积分器 (Integrators) 中的硬编码与循环冗余**
   * **反模式**：`DiffusionIntegrator` 和 `ElasticityIntegrator` 中到处散落着手写的 H1 物理梯度变换。例如 Elasticity 中冗长地手写了应变矩阵 `B` 的逐元素展开。
   * **后果**：代码冗长且容易出错。物理方程的组装与基函数的映射逻辑严重耦合。
3. **缺少全局拓扑方向 (Missing Edge Orientation)**
   * **反模式**：Nedelec 单元定义在边上（具有切向），相邻单元共享边时，局部方向可能相反。当前的 `FESpace` 只有自由度索引，完全缺失拓扑方向感知（Sign Flip）。
4. **测试脚本混乱冗长**
   * **反模式**：`tests/CMakeLists.txt` 像记流水账一样逐个列出十几个测试和依赖，极难维护。
5. **弯曲几何 (Curved Geometry) 潜在隐患**
   * 在二阶单元上，`Jacobian` 的计算必须精确基于二阶节点的几何形函数导数。如果直接用一阶节点坐标去乘高阶导数，弯曲几何的法向和雅可比将是错误的。

---

### 🛠️ 步骤化重构计划 (强制不向后兼容)

请按照以下顺序**逐块完成、逐块验证、立即提交**。

#### 步骤 1：测试脚本极简化与 CMake 瘦身
**目标**：解决测试文件编译子脚本混乱的问题，消除样板代码。
* **行动**：修改 `tests/CMakeLists.txt`。我们知道所有的核心模块最终都汇聚到 `mpfem::problem`。直接使用 `file(GLOB)` 或列表遍历批量生成测试。
```cmake
# tests/CMakeLists.txt 重构
file(GLOB TEST_SOURCES "test_*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # 统一链接最高层抽象 mpfem::problem 即可涵盖底层
    target_link_libraries(${test_name} PRIVATE mpfem::problem GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
```

#### 步骤 2：抽离 MapType 与集中化 Piola 变换
**目标**：彻底抹除 `ElementTransform::transformGradient`，实现真正的泛型算子。
1. 在 `src/fe/finite_element.hpp` 增加枚举：
   ```cpp
   enum class MapType { VALUE, COVARIANT_PIOLA, CONTRAVARIANT_PIOLA };
   virtual MapType mapType() const = 0; // FiniteElement 的纯虚函数
   ```
2. 建立新类 `ShapeEvaluator` (或直接集成在 `ReferenceElement` 中提供更高阶的求值方法)：
   * `evalPhysShape(trans, refShape, physShape)`
   * `evalPhysDeriv(trans, refDeriv, physDeriv)`
   * 内部通过 `switch(basis.mapType())` 进行**唯一一次**的 Piola 派发。
     * `VALUE`: 标量梯度乘 $J^{-T}$。
     * `COVARIANT_PIOLA` (ND): 基函数向量乘 $J^{-T}$，旋度乘 $\frac{1}{\det J} J$。
3. **修改积分器**：以 `DiffusionIntegrator` 为例，彻底删除三层 for 循环，重写为矩阵直接运算：
   ```cpp
   // 重构后的极其精简的组装
   const Matrix& physGrad = evaluator.getPhysDerivs(); // 内部已处理完 Piola
   elmat.noalias() += w * (physGrad * D * physGrad.transpose());
   ```

#### 步骤 3：FESpace 的方向感知 (Orientation) 注入
**目标**：为 Nedelec 单元提供边翻转系数 $\pm 1$。
1. **Mesh 层**：在 `buildTopology` 中确定**标准全局边方向**（始终由全局 Index 小的顶点指向大的顶点）。
2. **FESpace 层**：分配 DOF 时，若 `fec_->basisType() == BasisType::ND`，则调用 `Mesh::getElementEdges` 获取各边。
   * 比对局部边定义（如局部节点 $v_0 \to v_1$）与全局边定义。
   * 若全局边也是 $v_0 \to v_1$（即 $v_0 < v_1$），记录此 DOF 的 Orientation 为 `1`，反之记录为 `-1`。
3. 暴露接口 `std::span<const int> getElementOrientations(elemIdx)`。组装时将其传给 `ShapeEvaluator`，在输出物理域形函数前直接与数组对应元素相乘。

#### 步骤 4：基于拓扑的 Nedelec 基函数动态生成
**目标**：杜绝手写数学长式，运用质心坐标 (Barycentric) 算法生成 ND 单元。
* 修改 `src/fe/nd.cpp`。不再硬编码每个边，而是：
  ```cpp
  // 伪代码：在 evalShape / evalDerivatives 中
  Barycentric b(geom_, xi);
  for (int e = 0; e < numEdges; ++e) {
      auto [i, j] = geom::edgeVertices(geom_, e);
      // N_e = L_i * grad(L_j) - L_j * grad(L_i)
      for(int d=0; d<dim; ++d) {
          shape(e, d) = b.L[i] * b.dL[j][d] - b.L[j] * b.dL[i][d];
      }
      // Curl N_e = 2 * grad(L_i) x grad(L_j)
      // (在此处填入计算 3D 向量叉乘的极简代码)
  }
  ```
* 对于二阶 Nedelec，同理结合边中点的权重按算法展开，代码量可控制在30行以内。

#### 步骤 5：核查弯曲几何 (Curved Element) 修复
**目标**：确保二阶雅可比计算无误。
* 检查 `src/fe/element_transform.cpp` 中的 `computeJacobian()`。
* **修复要点**：
  ```cpp
  // 旧的可能有问题的逻辑：假定全部是顶点
  // 必须确保 nodesBuf_ 包含了高阶节点，并且调用的是：
  GeometryMapping::evalDerivatives(geometry_, geomOrder_, ipXi_, geoShapeDerivatives_);
  jacobian_.setZero();
  for (int i = 0; i < numNodes_; ++i) {
      // 如果弯曲几何，这里必须循环到所有 10 个(四面体) 或 27 个(六面体)节点
      jacobian_ += nodesBuf_[i] * geoShapeDerivatives_.row(i); 
  }
  ```

#### 步骤 6：严格克罗内克与切向连续性测试 (`test_nd.cpp`)
**目标**：用数学验证设计。
1. **Kronecker Test**：
   * 测试在一阶 Tetrahedron 的 6 条参考边上用高斯数值积分计算 $\int_{e_j} \hat{\mathbf{N}}_i \cdot \mathbf{t}_j dl$。
   * 用 `EXPECT_NEAR(matrix(i, j), (i==j? 1.0 : 0.0), 1e-10)` 验证。
2. **Patch Test**：
   * 构造包含两个共享曲面（二阶几何坐标扭曲）四面体的 Mesh。
   * 获取共用边上的物理域形函数值 $\mathbf{N}_{phys}^{(1)}$ 和 $\mathbf{N}_{phys}^{(2)}$。
   * 验证 `(Orientation1 * N1 + Orientation2 * N2) · Tangent == 0`，证明切向方向实现了完全连续且法向允许不连续。
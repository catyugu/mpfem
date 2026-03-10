# 项目：mpfem

## 宪法

* 语言要求：现代C++为主
* 构建系统：cmake，多级构建（库、可执行文件、测试）
* 构建目标：为多物理场仿真库（而不是可执行文件，可执行文件应各自独立编译为example或test目标）
* 测试要求：基于案例的测试
* 能复用的代码尽可能不从头重写。
* 保证不同环节（单元定义、组装、强制边界条件施加）代码的解耦和可复用性。
* 代码尽可能简洁、清晰、通用、高效，兼顾可读性。
* 保证依赖倒置原则，避免面条代码。
* 保证前后端的解耦，即从材料+物理场到Coefficient+Assembler的转换。
* 必须使用日志系统而非原始的cout等。

## 目标

* 需要支持一阶、二阶（弯曲单元）的单元形函数。
* 支持各种不同形状的单元（对三角、四边形、四面体、六面体、棱柱、金字塔形的支持）。
* 实现节点元（H1）（标量和矢量（用于位移场求解））、棱边元（ND）、通量单元（RT）、离散（L2）单元的集成。
* 实现稳态（线性，非线性）、瞬态、特征值等问题求解。
* 支持直接法、迭代法求解，使用策略模式以便以统一格式接入不同的求解器。
* 实现基本的有限元静电学、热学、刚体力学仿真。
* 实现多物理场计算，如焦耳热、热膨胀效应。
* 架构必须可扩展到更多不同的物理场和多物理场。
* 基于XML配置文件的仿真配置，支持变量系统和表达式。
* 需要有接口能解析来自COMSOL的网格文件、材料属性文件
* 需要有接口能生成和COMSOL兼容的结果文件（用于验证结果）以及用于可视化的vtu文件（或者文件夹）。

## 可能有用的外部库(如果本地没有安装，可以从github直接拉)

* CPM：用于cmake依赖管理
* MFEM：一个成熟的有限元库，有内置的网格管理，组装，求解等。供参考，不要引入编译。
* GTest：用于单元测试
* Eigen：可能用于小规模的矩阵求解。
* MKL PARDISO：用于稀疏矩阵方程的快速求解。
* 其他可能有用的依赖用于JSON，XML等配置文件解析、表达式解析等。

## 要求

* 实现并积极使用一个极简的线程安全的日志系统，便于后续调试。
* 详尽的文档记录，统一在doc/目录下，有序存放程序架构，计划，TODO等。
* 深入学习mfem的架构和实现，保证程序的效率、可扩展性。
* 如有必要，可以直接抄袭/或者改写mfem的源码。
* 尽可能避免简化或者占位实现，保证程序的良好设计。对实现存在简化或者只是占位实现的地方，请加入TODO注释以说明。
* 注意：需要考虑区分内边界（被多个元素共享）和外边界。内边界默认情况下（除非被指定）不应该施加任何处理。

## 要求的案例

你的完成度需要至少达到完成我给出的案例。

案例说明如下：

* 变量表

```text
L	9[cm]	0.09 m	Length
rad_1	6[mm]	0.006 m	Bolt radius
tbb	5[mm]	0.005 m	Thickness
wbb	5[cm]	0.05 m	Width
mh	3[mm]	0.003 m	Maximum element size
htc	5[W/m^2/K]	5 W/(m²·K)	Heat transfer coefficient
Vtot	20[mV]	0.02 V	Applied voltage

```

## 一阶测试：`cases/busbar/`下

* 案例描述：一个母线板，上面有三个螺丝。求解稳态问题。
* 网格文件：mesh.mphtxt
* 参考求解配置文件的形态：case.xml
* 一共7个domain，43个boundary。（到此你应该检验网格读入是否正确）
* 三个物理场：电势、固体传热、刚体力学。
* 材料：Copper用于domain 1（板体），Titanium beta-21S用于domain 2-7，材料属性见于material.xml
* 边界条件：
  * 电势：boundary 43电势为Vtot，boundary 8和15电势为0（接地）。其余外边界绝缘。
  * 固体传热：boundary 1-7, 9-14, 16-42有对流换热边界，换热系数为htc，外界温度为293.15K，其余外边界热绝缘。
  * 刚体力学：boundary 8, 15, 43 为固定（零位移）。
* 考虑焦耳热和热应变。
* COMSOL结果文件：result.txt。
* 请将我们的结果和COMSOL结果文件进行对比。（这一步可以用Python等外部脚本完成，C++代码只需要能export标准格式的结果即可）

## 二阶测试：`cases/busbar_order2/`下

* 其余与一阶情况一样，只是网格为二阶，基函数也为二阶多项式

## 可以参考的设计

### 第一层：几何与拓扑层 (Geometry & Topology)

这一层**只**关心网格长什么样，实体之间如何连接。它对物理场和微积分一无所知。

* **核心模块：`Mesh` / `Triangulation**`
* **职责：** 存储节点坐标，维护基于 CSR 格式的实体连通性图，管理自适应网格细化（AMR）的树状结构，以及边界标记（Boundary ID）。

```cpp
class Mesh {
public:
    // 获取实体连通性 (如 面找单元 d1=2, d2=3)
    std::span<const int> get_connectivity(int d1, int d2, int entity_id) const;
    
    // 获取给定实体的所有顶点坐标 (用于后续的雅可比计算)
    std::vector<Point<dim>> get_vertex_coordinates(int cell_id) const;
    
    // 获取边界标记 (区分不同边界条件)
    int get_boundary_id(int face_id) const;
};

```

### 第二层：有限元空间层 (Finite Element Spaces)

这一层是纯粹的数学抽象。它定义了参考元上的多项式及其导数。为了支持**多变量 $H^1$ 单元（如二维形变场 $\mathbf{u} = [u_x, u_y]^T$）**，我们不仅要有标量单元，还要有向量单元抽象。

* **核心模块：`FiniteElement` (基类) 与 `FESystem` (耦合类)**
* **职责：** 提供参考元上的形函数值和梯度。
* **多变量支持：** 通过 `n_components` 属性。形变场是一个 `n_components = dim` 的向量单元。

```cpp
class FiniteElement {
protected:
    int degree;           // 阶数 (1代表线性, 2代表二阶弯曲/二次插值)
    int n_components;     // 物理量分量数 (温度=1, 2D位移=2)
    int dofs_per_cell;    // 单元内部的自由度总数
    
public:
    virtual ~FiniteElement() = default;

    // 获取在参考元高斯点 q 上，第 i 个形函数的具体分量 component 的值和梯度
    virtual double shape_value(int i, int q, int component = 0) const = 0;
    virtual Tensor<1, dim> shape_grad(int i, int q, int component = 0) const = 0;
};

// 派生类实现：线性与二次 Lagrange 单元
class FE_Q : public FiniteElement { /* 实现了高斯洛巴托节点的插值多项式 */ };

```

### 第三层：自由度分发层 (DoF Management)

这是整个解耦架构的**灵魂**。它斩断了“网格节点”与“矩阵行号”的硬性绑定。

* **核心模块：`DoFHandler**`
* **职责：** 将 `FiniteElement` 映射到 `Mesh` 上，为每一个自由度分配一个全局唯一的整数 ID。处理多场耦合时，它会将电场、温度场、形变场的自由度混合打包或分块。

```cpp
class DoFHandler {
private:
    const Mesh* mesh;
    const FiniteElement* fe;
    std::vector<int> cell_dof_indices; // 一维展平的局部到全局映射数组

public:
    void distribute_dofs(); // 核心：遍历网格，分配全局 ID
    int n_dofs() const;     // 返回系统总自由度数 (即全局矩阵的大小)
    
    // 获取某个单元对应的所有全局自由度索引
    std::vector<int> get_local_to_global_mapping(int cell_id) const;
};

```

### 第四层：装配引擎层 (Assembly & FEValues)

这一层连接了抽象数学和物理现实。物理学家（用户）在这里写下变分方程。

* **核心模块：`FEValues` (微积分求值器) 与 `Assembler**`
* **职责：** `FEValues` 负责在循环中执行**参考元到物理元的映射**（链式法则、雅可比行列式计算）。它处理了高阶弯曲单元引发的所有几何复杂性。

```cpp
class FEValues {
public:
    // 在每个单元循环开始时调用，更新当前物理单元上的雅可比、形函数梯度等
    void reinit(int cell_id); 
    
    // 获取当前物理高斯点上的权值 (det(J) * w_q)
    double JxW(int q) const;
    
    // 获取物理空间下的基函数梯度: J^{-T} * \nabla_{\xi} \hat{N}
    Tensor<1, dim> shape_grad(int i, int q, int component = 0) const;

    // 【极其重要的非线性功能】：将上一迭代步的全局解 X 投影到当前高斯点上
    // 用于计算如非线性应力 \sigma(u), 变电导率 k(T) 等
    void get_function_values(const Vector& global_X, std::vector<double>& local_q_values) const;
    void get_function_gradients(const Vector& global_X, std::vector<Tensor<1, dim>>& local_q_grads) const;
};

```

### 第五层：系统求解层 (Linear/Nonlinear Solvers & Time Stepping)

为了支持非线性和瞬态问题，我们彻底放弃显式组装刚度矩阵 $K$ 的陈旧观念，转而采用**基于残差 (Residual-based)** 的非线性迭代架构。

* **核心模块：`NonlinearProblem` 与 `TimeIntegrator**`
* **职责：** 瞬态问题（如隐式 Euler 或 Newmark-$\beta$）会转换为一系列非线性稳态问题。非线性求解器（如 Picard）只需向用户索要残差。

```cpp
class NonlinearSystem {
public:
    SparseMatrix Jacobian;
    Vector Residual;
    Vector Solution;

    // 用户只需提供计算残差和雅可比的 Lambda 函数或回调
    virtual void build_residual() = 0; 
    virtual void build_jacobian() = 0;
    
    // 处理 Dirichlet 边界条件：通常采用矩阵对角线置 1，右端项置为边界值的代数操作，或者通过罚函数法 (Penalty Method)
    void apply_dirichlet_bcs(const std::map<int, double>& boundary_values);
};

```

### 第六层：物理抽象层 (用户接口)

有了底层强大的支撑，处理一个**大变形弹性力学（形变场 $\mathbf{u}$，多变量 $H^1$ 单元，非线性）**的代码在用户看来极其简洁：

```cpp
// 用户的组装逻辑 (伪代码展示灵活性)
void build_residual() {
    for (int cell = 0; cell < mesh.n_cells(); ++cell) {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(Solution, grad_u); // 获取上一步的位移梯度

        for (int q = 0; q < n_q_points; ++q) {
            // 计算非线性物理量，例如基于变形梯度的第二 Piola-Kirchhoff 应力
            Tensor<2, dim> stress = compute_nonlinear_stress(grad_u[q]); 
            
            for (int i = 0; i < dofs_per_cell; ++i) {
                // 组装残差 (弱形式的纯翻译)
                local_residual[i] -= (stress * fe_values.shape_grad(i, q)) * fe_values.JxW(q);
            }
        }
        // 将局部矩阵根据 DoFHandler 映射累加到全局稀疏矩阵
        distribute_local_to_global(); 
    }
}

```

### 为什么这套架构完美适应你的需求？

1. **二阶/线性随意切换：** 只需要在初始化时传入 `FE_Q<2>` 还是 `FE_Q<1>`，底层的 `FEValues` 会自动抽取不同的高斯积分规则，组装代码**一行都不用改**。
2. **多分量形变场支持：** 通过 `FEValues::shape_grad(i, q, component)`，矢量场（力、位移）的张量运算与标量场完全统一。
3. **非线性与瞬态灵活：** `get_function_gradients` 将历史时间步或上一迭代步的解优雅地投影回积分点，极其方便地处理随场变化的本构关系。边界条件（Neumann）只需在表面积分循环中增加一项即可。

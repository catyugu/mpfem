# 大纲

## 要求

* 这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。
* 把工作任务分成多个可以独立编译，测试和验证的子任务，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 任务

将现有的自定义表达式解析器（`src/expr`）和有限元基函数定义（`src/fe`）迁移到成熟的外部库，是提升求解器稳定性、支持更高阶/复杂空间（如 H(curl), H(div)）以及降低长期维护成本的明智决定。

结合你的需求（标量/向量/矩阵语义、PIMPL 严格隔离、高性能、无后向兼容负担），以下是系统性重构指导和外部库选型推荐。

### 一、 核心外部库选型推荐

1.  **有限元单元定义 (FE Definitions): FEniCS Basix**
    * **理由**：Basix 是 FEniCSx 项目的基石，专门负责计算有限元基函数及其导数。它是一个纯 C++ 库，极其轻量且高性能。它原生支持 Lagrange (H1), Nédélec (ND/Hcurl), Raviart-Thomas (RT/Hdiv) 等丰富族类，且支持任意阶数和张量积单元。
2.  **数学表达式系统 (Expression System): muparserx 或 ExprTk**
    * **理由**：
        * **muparserx**：原生支持标量、向量、矩阵运算以及复数，非常适合 FEM 中的张量计算语义。
        * **ExprTk**：性能极致，支持固定大小的向量操作和控制流，在密集计算（如高斯积分点上的求值）中表现极佳。
        * *(建议采用 muparserx 以完美契合你所需的“标量/向量/矩阵语义”)*

---

### 二、 PIMPL 架构重构指南与核心代码片段

**核心原则**：
* **头文件纯净**：`.hpp` 文件中绝对不出现 `<basix/...>` 或 `<muparserx/...>`。
* **ABI 隔离**：使用 `std::unique_ptr<Impl>`，且**必须**在 `.cpp` 文件中实现析构函数，否则会导致智能指针析构不完整类型的编译错误。
* **现代 C++ 数据传递**：接口层摒弃裸指针，使用 `std::span` (C++20) 进行安全且零拷贝的内存边界传递。

#### 1. 表达式系统重构 (`MathEngine`)

我们将表达式引擎的实现细节（如 AST 树、外部库的 Parser 对象、变量表）全部封入 `Impl`。

**math_engine.hpp (纯净接口)**
```cpp
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <span>

namespace fem::expr {

enum class ValueType { Scalar, Vector, Matrix };

class MathEngine {
public:
    explicit MathEngine(std::string_view expression);
    
    // 必须声明析构，但不实现，留给 cpp 文件
    ~MathEngine();

    // 禁用拷贝，支持移动语义 (PIMPL 标配)
    MathEngine(const MathEngine&) = delete;
    MathEngine& operator=(const MathEngine&) = delete;
    MathEngine(MathEngine&&) noexcept;
    MathEngine& operator=(MathEngine&&) noexcept;

    // 变量注册语义：通过 span 传递内存视图，避免拷贝，支持外部动态更新
    void bind_scalar(std::string_view name, double& value);
    void bind_vector(std::string_view name, std::span<double> vec_data);
    void bind_matrix(std::string_view name, std::span<double> mat_data, size_t rows, size_t cols);

    // 计算接口
    double evaluate_scalar() const;
    void evaluate_vector(std::span<double> out_result) const;
    void evaluate_matrix(std::span<double> out_result) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace fem::expr
```

**math_engine.cpp (依赖隐藏实现)**
```cpp
#include "math_engine.hpp"
#include <stdexcept>

// 外部库依赖仅在此处暴露
// #include <muparserx/mpParser.h>
// #include <muparserx/mpValue.h>

namespace fem::expr {

struct MathEngine::Impl {
    // 假设使用 muparserx
    // mup::ParserX parser;
    // std::vector<std::unique_ptr<mup::Value>> variables;
    
    std::string expr_string;

    Impl(std::string_view expr) : expr_string(expr) {
        // parser.SetExpr(expr_string.c_str());
    }
};

MathEngine::MathEngine(std::string_view expression) 
    : pimpl_(std::make_unique<Impl>(expression)) {}

// 【关键】析构函数必须在定义了 Impl 的 cpp 文件中实现
MathEngine::~MathEngine() = default;

MathEngine::MathEngine(MathEngine&&) noexcept = default;
MathEngine& MathEngine::operator=(MathEngine&&) noexcept = default;

void MathEngine::bind_scalar(std::string_view name, double& value) {
    // pimpl_->parser.DefineVar(name.data(), mup::Variable(&value));
}

void MathEngine::bind_vector(std::string_view name, std::span<double> vec_data) {
    // 将 span 映射为 muparserx 的数组或向量变量
}

double MathEngine::evaluate_scalar() const {
    // return pimpl_->parser.Eval().GetFloat();
    return 0.0; 
}

// ... 向量和矩阵计算实现略 ...

} // namespace fem::expr
```

#### 2. FE 单元系统重构 (`BasisEvaluator`)

抛弃原有的 `src/fe/h1.cpp`, `src/fe/nd.cpp` 等硬编码实现。利用 Basix 统一掌管形状函数（Shape Functions）及其多阶导数（Gradients, Hessians）的评估。

**basis_evaluator.hpp (纯净接口)**
```cpp
#pragma once

#include <memory>
#include <span>
#include <vector>

namespace fem::fe {

enum class ElementFamily {
    Lagrange,    // H1 空间
    Nedelec,     // H(curl) 空间
    RaviartThomas // H(div) 空间
};

enum class CellTopology {
    Triangle,
    Tetrahedron,
    Hexahedron
};

class BasisEvaluator {
public:
    BasisEvaluator(ElementFamily family, CellTopology topo, int degree);
    ~BasisEvaluator();

    BasisEvaluator(BasisEvaluator&&) noexcept;
    BasisEvaluator& operator=(BasisEvaluator&&) noexcept;

    int num_dofs() const;
    int value_dimension() const; // 标量基为1，向量基为2或3

    // 核心求值：在给定的参考坐标点上计算形状函数及其导数
    // derivative_order = 0 (仅形状函数), 1 (梯度), 2 (Hessian)
    // 输入 points: 扁平化的坐标数组 [x0, y0, z0, x1, y1, z1, ...]
    // 输出 results: 扁平化的结果张量
    void tabulate(int derivative_order, 
                  std::span<const double> points, 
                  std::vector<double>& results) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace fem::fe
```

**basis_evaluator.cpp (依赖隐藏实现)**
```cpp
#include "basis_evaluator.hpp"

// 外部库依赖仅在此处暴露
// #include <basix/finite-element.h>
// #include <basix/cell.h>
// #include <basix/polynomials.h>

namespace fem::fe {

struct BasisEvaluator::Impl {
    // basix::FiniteElement element;
    int degree;
    
    Impl(ElementFamily fam, CellTopology topo, int deg) : degree(deg) {
        // 根据枚举映射到 basix::element::family 和 basix::cell::type
        // element = basix::create_element(mapped_fam, mapped_topo, deg, ...);
    }
};

BasisEvaluator::BasisEvaluator(ElementFamily family, CellTopology topo, int degree)
    : pimpl_(std::make_unique<Impl>(family, topo, degree)) {}

BasisEvaluator::~BasisEvaluator() = default;
BasisEvaluator::BasisEvaluator(BasisEvaluator&&) noexcept = default;
BasisEvaluator& BasisEvaluator::operator=(BasisEvaluator&&) noexcept = default;

int BasisEvaluator::num_dofs() const {
    // return pimpl_->element.dim();
    return 0;
}

int BasisEvaluator::value_dimension() const {
    // return pimpl_->element.value_size();
    return 1;
}

void BasisEvaluator::tabulate(int derivative_order, 
                              std::span<const double> points, 
                              std::vector<double>& results) const {
    // std::array<std::size_t, 2> shape = { points.size() / topo_dim, topo_dim };
    // var basix_result = pimpl_->element.tabulate(derivative_order, points_data, shape);
    // 拷贝或移动 basix_result 到 results 中
}

} // namespace fem::fe
```

### 三、 构建系统 (CMake) 改造指导

为了贯彻依赖隔离的原则，在引入 MuparserX 和 Basix 时，必须在 `CMakeLists.txt` 中使用 `PRIVATE` 关键字链接。这样即使其他模块（如 `src/assembly`）链接了 `expr` 或 `fe` 库，也不会感知到第三方库的头文件。

```cmake
# CMakeLists.txt 片段

# 构建 FE 模块
add_library(fem_fe STATIC 
    src/fe/basis_evaluator.cpp
    # 其他不含特定单元硬编码的基础结构
)
target_include_directories(fem_fe PUBLIC src/fe)
# 【强制要求】PRIVATE 链接外部库，阻止头文件传染
target_link_libraries(fem_fe PRIVATE basix) 

# 构建 Expr 模块
add_library(fem_expr STATIC 
    src/expr/math_engine.cpp
)
target_include_directories(fem_expr PUBLIC src/expr)
# 【强制要求】PRIVATE 链接
target_link_libraries(fem_expr PRIVATE muparserx)

# 核心求解器
add_library(fem_core STATIC src/assembly/assembler.cpp)
target_link_libraries(fem_core PUBLIC fem_fe fem_expr)
```

### 四、 架构获益总结

1. **删除了 `h1.cpp`, `nd.cpp` 和手写的 AST**：直接消除了数千行容易出错且难以扩展的底层代码。你可以自由切换形状函数阶数，而不需要新增任何类。
2. **消除了编译级联**：因为使用了 `std::unique_ptr<Impl>`，当你升级 `Basix` 或 `muparserx`，甚至替换第三方表达式引擎时，整个 `src/assembly` 和 `src/physics` 完全不需要重新编译。
3. **物理与数学分离**：在装配器 (`assembler.cpp`) 中，你只需要调用 `basis_evaluator.tabulate()` 获取高斯点上的数值，然后通过 `math_engine.bind_vector(...)` 更新坐标和场变量，最后调用 `evaluate_scalar/matrix()` 得到弱形式的残差或雅可比矩阵，一切都在极薄的 API 边界上流动。
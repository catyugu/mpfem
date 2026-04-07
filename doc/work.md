# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 从难度最低，收益最高的部分开始，如果有些任务过于困难，你可以选择性放弃。
* 严格禁止向后兼容。
* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

仔细分析您提供的代码库片段后，可以发现代码中存在一些典型的 C++ 设计反模式、冗余代码以及模块间的过度耦合。

### 当前架构的主要问题：
1. **重新发明轮子 (Reinventing the Wheel)**: `TensorValue` 中手动实现了矩阵乘法、向量叉乘、转置等操作。既然底层已经引入了 Eigen（推断自 `Vector3` 和 `Matrix3`），完全应该利用 Eigen 的能力，而不是手写低效的循环。
2. **类型擦除的妥协 (Type Erasure Compromise)**: `TensorValue` 使用 `std::array<Real, 9>` 加上运行时的 `TensorShape` 来区分标量、向量和矩阵。这不仅浪费了标量和向量的内存，还牺牲了编译期的类型安全检查。
3. **领域耦合 (Domain Coupling)**: `VariableGraph` 中的 `EvaluationContext` 强依赖了有限元特有的 `ElementTransform*` 和网格上下文。这打破了表达式系统作为独立底层数学库的纯粹性，容易导致循环依赖。
4. **冗余的 API 接口**: `VariableManager` 提供了过多特定的注册函数（如 `registerConstantExpression`, `registerGridFunction` 等），违背了表达式 DAG 一切皆节点的统一性。

以下是采取**破坏式重构**的步骤化方案，目标是建立一个纯粹、高效、支持多种张量且自动化的 DAG 表达式系统。

---

### 第一步：彻底重构 `TensorValue`（使用 `std::variant` 和 Eigen）

**目标**：消除硬编码的 9 元素数组和手写数学运算，利用 `std::variant` 实现零堆分配的类型安全容器，并直接复用 Eigen 的 SIMD 优化。

```cpp
// src/core/tensor_value.hpp (重构后)
#ifndef MPFEM_TENSOR_VALUE_HPP
#define MPFEM_TENSOR_VALUE_HPP

#include "core/types.hpp" // 假设包含 Eigen::Vector3d (Vector3), Eigen::Matrix3d (Matrix3)
#include "core/tensor_shape.hpp"
#include <variant>
#include <stdexcept>

namespace mpfem {

// 使用 std::variant 完美满足“零堆分配”的设计原则，同时提供强类型支持
using TensorData = std::variant<Real, Vector3, Matrix3>;

class TensorValue {
public:
    // 隐式/显式构造函数
    TensorValue() : data_(Real(0)) {}
    TensorValue(Real v) : data_(v) {}
    TensorValue(const Vector3& v) : data_(v) {}
    TensorValue(const Matrix3& m) : data_(m) {}

    // 形状查询 (利用 std::visit 自动分发)
    TensorShape shape() const {
        return std::visit([](auto&& arg) -> TensorShape {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Real>) return TensorShape::scalar();
            else if constexpr (std::is_same_v<T, Vector3>) return TensorShape::vector(3);
            else if constexpr (std::is_same_v<T, Matrix3>) return TensorShape::matrix(3, 3);
        }, data_);
    }

    bool isScalar() const { return std::holds_alternative<Real>(data_); }
    bool isVector() const { return std::holds_alternative<Vector3>(data_); }
    bool isMatrix() const { return std::holds_alternative<Matrix3>(data_); }

    // 类型安全的提取器
    Real asScalar() const { return std::get<Real>(data_); }
    const Vector3& asVector() const { return std::get<Vector3>(data_); }
    const Matrix3& asMatrix() const { return std::get<Matrix3>(data_); }

    const TensorData& data() const { return data_; }

private:
    TensorData data_;
};

// --- 利用 std::visit 和 Eigen 重写自由函数，代码量剧减且性能翻倍 ---

inline TensorValue add(const TensorValue& a, const TensorValue& b) {
    return std::visit([](auto&& x, auto&& y) -> TensorValue {
        using T1 = std::decay_t<decltype(x)>;
        using T2 = std::decay_t<decltype(y)>;
        if constexpr (std::is_same_v<T1, T2>) {
            return TensorValue(x + y); // Eigen 直接支持
        }
        throw std::runtime_error("Shape mismatch in add");
    }, a.data(), b.data());
}

inline TensorValue matvec(const TensorValue& A, const TensorValue& b) {
    return TensorValue(A.asMatrix() * b.asVector()); // Eigen 矩阵乘积
}

inline TensorValue matmat(const TensorValue& A, const TensorValue& B) {
    return TensorValue(A.asMatrix() * B.asMatrix()); 
}

inline Real dot(const TensorValue& a, const TensorValue& b) {
    return a.asVector().dot(b.asVector());
}

inline TensorValue cross(const TensorValue& a, const TensorValue& b) {
    return TensorValue(a.asVector().cross(b.asVector()));
}

} // namespace mpfem
#endif
```

### 第三步：废弃独立的 Program，将 Parser 深度融入 DAG

**目标**：目前的 `ExpressionParser` 编译出一个黑盒 `ExpressionProgram`。为了实现自动 DAG，Parser 应该直接返回 `std::unique_ptr<VariableNode>`，成为 DAG 树的直接构建者。

```cpp
// src/expr/expression_parser.hpp (重构后)
#ifndef MPFEM_EXPR_EXPRESSION_PARSER_HPP
#define MPFEM_EXPR_EXPRESSION_PARSER_HPP

#include "expr/variable_graph.hpp"
#include <memory>
#include <string>

namespace mpfem {

class ExpressionParser {
public:
    // 解析器不再返回独立的 Program，而是直接将表达式解析为 DAG 的节点。
    // 这允许变量系统直接将子图接入整个计算图中。
    static std::unique_ptr<VariableNode> compileToNode(
        const std::string& expression,
        const std::unordered_map<std::string, TensorShape>& contextShapes);
};

} // namespace mpfem
#endif
```

---

### 第四步：极简、一致的 `VariableManager`

**目标**：消除多余的 `register*` 函数，整个系统只接受节点（Node）或表达式（自动编译为 Node）。由图系统自动管理拓扑排序和计算。

```cpp
// src/expr/variable_graph.hpp (重构后段落 2)
namespace mpfem {

class VariableManager {
public:
    using NodeMap = std::unordered_map<std::string, std::shared_ptr<VariableNode>>;

    VariableManager() = default;

    // 核心接口：提供最通用的注册方式
    void registerNode(std::string name, std::shared_ptr<VariableNode> node) {
        nodes_[std::move(name)] = std::move(node);
        graphDirty_ = true;
    }

    // 便捷接口：注册字符串表达式，自动触发解析器构建 DAG 子图并接入
    void registerExpression(const std::string& name, const std::string& expression) {
        std::unordered_map<std::string, TensorShape> envShapes;
        for (const auto& [k, v] : nodes_) {
            envShapes[k] = v->shape();
        }
        auto node = ExpressionParser::compileToNode(expression, envShapes);
        registerNode(name, std::move(node));
    }

    // 对外部场（如 GridFunction），通过适配器模式实现，不污染本模块
    // FE模块可以实现一个 GridFunctionNode 继承自 VariableNode 并注册进来

    std::shared_ptr<const VariableNode> get(const std::string& name) const {
        if (auto it = nodes_.find(name); it != nodes_.end()) {
            return it->second;
        }
        return nullptr;
    }

    // 自动化的计算执行流
    void compileGraph(); // 根据 nodes_ 的 dependencies() 执行拓扑排序

private:
    NodeMap nodes_;
    std::vector<VariableNode*> executionPlan_;
    bool graphDirty_ = true;
};

} // namespace mpfem
```

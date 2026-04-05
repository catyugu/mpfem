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
* 删除冗余的成员变量、接口等。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

从你提供的代码来看，系统正处于从**传统面向对象的回调式解析（Coefficient体系）**向**现代基于DAG（有向无环图）的计算图解析（Variable体系）**过渡的尴尬阶段。这两者在设计哲学上是冲突的，导致了代码中存在明显的失配和性能隐患。 宁可采取破坏式重构而非向后兼容，意图是**完全用自动DAG解析取代原始系统**，以下是代码中存在的明显反模式（Anti-patterns）以及大刀阔斧的重构建议。

### 🚨 核心反模式 (Anti-Patterns) 诊断

#### 1. 运行时字符串求值与动态解析（致命性能损耗）
* **问题位置**：`GraphExternalSymbolResolver` 的签名设计为 `std::function<bool(std::string_view, ElementTransform&, Real, double&)>`。
* **反模式**：在求解偏微分方程（FEM）时，求值函数会被在**每一个单元的每一个积分点**上调用数百万次。如果在热点循环（Tight Loop）中通过字符串 (`std::string_view`) 去动态查找变量状态，这会带来灾难性的性能开销。
* **重构方向**：**编译期绑定（AOT Binding）**取代**运行时查找（Runtime Lookup）**。变量的内存地址偏移或数据指针必须在 `compileGraph()` 阶段就确定好，求值时只读内存。

#### 2. 深层虚函数与 `std::function` 嵌套的逐点求值 (Point-wise Evaluation)
* **问题位置**：`fe/coefficient.hpp` 中的 `FunctionCoefficient` 包装了 `std::function<void(ElementTransform&, Real&, Real)>`。
* **反模式**：这是典型的旧式 FEM 库设计。为了应对多变的物理场参数，使用了最昂贵的抽象。一次积分点计算 = 1次虚函数调用 (`eval`) + 1次 `std::function` 堆栈调用。无法利用 SIMD 和缓存局部性。
* **重构方向**：**向量化/批量求值（Vectorized/Batched Evaluation）**。直接废弃 `Coefficient` 类，使用 `VariableNode` 的 `std::span<double>` 批量求值接口。

#### 3. 冗余的适配器桥接（Adapter Bloat）
* **问题位置**：`expression_coefficient_factory.hpp`。
* **反模式**：试图把现代的字符串表达式解析器塞进陈旧的 `Coefficient` 壳子里。这保留了旧系统的所有性能缺点，同时又增加了新系统的复杂性。
* **重构方向**：直接**删除** `fe/coefficient.hpp` 和 `expression_coefficient_factory.hpp`。让所有物理场、积分器 (Integrators) 直接持有 `VariableNode*` 或直接从 `EvaluationContext` 读取数据。

#### 4. 全局单例 (Global Singleton) 滥用
* **问题位置**：`ExpressionParser::instance()`。
* **反模式**：表达式解析器被设计成了单例。这不仅破坏了线程安全（多线程并发编译表达式时可能加锁或崩溃），还阻碍了上下文隔离。
* **重构方向**：将 `ExpressionParser` 实例绑定到 `VariableManager` 中，或者作为普通的纯局部工具类实例化。

---

### 🛠️ 破坏性重构方案 (Architectural Redesign)

#### 步骤一：彻底删除旧的 Coefficient 体系
删除 `fe/coefficient.hpp`、`expression_coefficient_factory.hpp`。
FEM 中的有限元组装器 (Assemblers) 或积分器 (Integrators) 不再接收 `std::unique_ptr<Coefficient>`。它们应该直接通过名称或 ID 向 `VariableManager` 请求计算好的数据。

#### 步骤二：改造 VariableGraph 采用“数据驱动”的设计
将 `VariableManager` 作为核心的数据总线。所有的外部状态（如坐标、时间、物理场U）在计算前**一次性推入/绑定**，而不是在求值时回调获取。

```cpp
// 建议的 variable_graph.hpp 重构

namespace mpfem {

// 1. 批量求值上下文（完全数据驱动，不包含任何回调）
struct EvaluationContext {
    double time = 0.0;
    // 使用 SOA (Structure of Arrays) 或扁平连续内存，便于 SIMD
    std::span<const double> x_coords; 
    std::span<const double> y_coords;
    std::span<const double> z_coords;
    // 单元相关数据
    std::span<const int> element_ids; 
    std::span<const int> domain_ids;
    
    // 外部提供的场变量 (例如: 温度场, 位移场)，在组装前预先提取好并传入
    std::unordered_map<std::string_view, std::span<const double>> external_fields;

    // 批量大小
    size_t batch_size() const { return x_coords.size(); }
};

// 2. 节点设计：只负责批量计算
class VariableNode {
public:
    virtual ~VariableNode() = default;

    virtual VariableShape shape() const = 0;
    // 注意：不再需要 stateTag() 和 isConstant()！
    // DAG 本身知道谁依赖谁。如果常数节点不更新，其输出缓存自然不变。

    // 核心接口：接收上下文，向 dest 输出计算结果。dest 预先分配好了 batch_size 大小。
    virtual void evaluate(const EvaluationContext& ctx, std::span<double> dest) const = 0;
    
    virtual std::vector<const VariableNode*> dependencies() const = 0;
};

class VariableManager {
public:
    // ...

    // 核心改变：编译期解析所有表达式并分配好内存空间
    void compileGraph();

    // 组装/积分阶段的核心调用
    // 给定上下文，批量计算所有节点。
    // 计算顺序由前序拓扑排序 (executionPlan_) 决定。
    void evaluateAll(const EvaluationContext& ctx);

    // 积分器直接获取某一个变量计算后的连续内存结果 (无需在循环内再次计算)
    std::span<const double> getEvaluatedData(std::string_view name) const;

private:
    // 扁平化数据存储：每个变量节点对应一块缓存内存
    std::unordered_map<const VariableNode*, std::vector<double>> node_buffers_;
    std::vector<const VariableNode*> executionPlan_;
};

}
```

#### 步骤三：消除运行时字符串回调 (Resolver 重构)
如果你需要外部符号（比如用户自定义表达式中写了 `sin(x) + Pressure`，`Pressure` 需要从别的物理场拿），千万不要在求值时查字典。

**旧做法（错误）：**
`sin(x) + resolver("Pressure", trans, ...)` 每次求值都触发 string 比较。

**新做法（正确）：**
在 `compileGraph` 时，如果遇到解析不了的外部变量 `Pressure`，将其转化为一个 **`ExternalInputNode`** 存入 DAG。
当调用 `evaluateAll(ctx)` 时，`ExternalInputNode` 会直接将 `ctx.external_fields["Pressure"]` 的数据 `memcpy`（或者通过指针引用）到自己的缓存中，下游的表达式节点直接从连续的数组缓存里读数据。

#### 步骤四：改造 ExpressionParser
取消单例。使用 JIT 或 AST 的解析器应该在绑定内存时直接操作数据指针或数组引用。

```cpp
// 取消单例
class ExpressionParser {
public:
    // 构造函数，不再是 instance()
    ExpressionParser() = default;

    // 变量绑定不再是双指针，而是批量求值的连续数组地址，或者偏移量
    struct VectorizedBinding {
        std::string name;
        const double* data_ptr; // 指向外部数组或 upstream 节点的 buffer
        size_t stride;
    };

    ScalarProgram compileScalarBatched(const std::string& expression,
                                       const std::vector<VectorizedBinding>& bindings);
};
```

### 🎯 重构后的实际应用流 (Workflow)

重构后，一次典型的 FEM 矩阵组装流程将变得极其简洁和高效：

1. **Setup Phase (初始化)**:
   用户输入配置: `Material_E = "2.0 * T + 100"`.
   `VariableManager` 解析表达式，建立 DAG: `T(External) -> Node(2.0*T) -> Node(+100) -> E(Output)`.
   执行 `compileGraph()`，拓扑排序确定执行顺序，并为每个节点分配 `batch_size`（比如一次计算一个 Element 的所有积分点，或者一次计算 1024 个积分点）大小的内存。

2. **Assembly Phase (紧凑循环 / 组装)**:
   ```cpp
   // 在积分器或组装器中：
   EvaluationContext ctx;
   ctx.x_coords = ...; // 填充当前批次的积分点坐标
   ctx.external_fields["T"] = ...; // 填入当前批次提取的温度场数据

   // 1. DAG 一键全量求值 (无任何字符串操作，无虚函数逐点调用，全是数组到数组的运算)
   var_manager.evaluateAll(ctx);

   // 2. 获取结果并进行矩阵组装
   std::span<const double> E_values = var_manager.getEvaluatedData("Material_E");
   
   for(size_t i = 0; i < ctx.batch_size(); ++i) {
       // 直接使用 E_values[i] 组装局部刚度矩阵 Ke
       // 完全移除了曾经在积分点循环里调用的 coef->eval() !
   }
   ```

### 总结
你现有的代码卡在“用面向对象的壳包装数据驱动的心”。打破向后兼容，**彻底抛弃 `fe/coefficient.hpp`**，将一切基于变量图的计算转化为**“批次输入 -> 图拓扑排序执行 -> 连续内存输出”**的数据流模型。这不仅将大幅减少抽象开销，还会让缓存命中率和执行效率实现质的飞跃。
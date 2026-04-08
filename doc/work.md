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
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

### 第一步：可验证的中间重构 (Phase 1: DAG 扁平化调度)

#### 1. 修改节点接口：显式传递批处理输入 (`src/expr/variable_graph.hpp`)
修改 `VariableNode` 的 `evaluateBatch` 签名，强制要求外界（Manager）把依赖项的批量计算结果喂给它，而不是让它自己去拿。

```cpp
// 在 src/expr/variable_graph.hpp 中：
namespace mpfem {

    class VariableNode {
    public:
        virtual ~VariableNode() = default;

        virtual TensorShape shape() const = 0;

        // 【修改点】：增加 inputs 参数。inputs 的大小等于依赖项的数量。
        // 每个 inputs[i] 是依赖项 d 对应的批量数据 span，大小为 dest.size()
        virtual void evaluateBatch(const EvaluationContext& ctx, 
                                   std::span<const std::span<const TensorValue>> inputs,
                                   std::span<TensorValue> dest) const = 0;

        virtual bool isConstant() const { return false; }
        virtual std::vector<const VariableNode*> dependencies() const { return {}; }
    };

    class VariableManager {
        // ... (保持其它 public 接口不变)
    private:
        // ...
        NodeStore nodes_;
        std::vector<const VariableNode*> executionPlan_;
        bool graphDirty_ = true;
    };
}
```

#### 2. 更新三个现有的节点实现 (`src/expr/variable_graph.cpp`)
这里我们将移除 `RuntimeExpressionNode` 中的隐式递归依赖计算（即将 `workspace.scratchpad` 的递归调用删掉），仅仅读取 `inputs`。

```cpp
// 在 src/expr/variable_graph.cpp 内部：

class RuntimeExpressionNode final : public VariableNode {
public:
    // 构造函数不变...

    // 【修改点】：直接使用 inputs
    void evaluateBatch(const EvaluationContext& ctx, 
                       std::span<const std::span<const TensorValue>> inputs, 
                       std::span<TensorValue> dest) const override
    {
        const size_t n = dest.size();
        const size_t m = dependencies_.size();

        // 不再需要 scratchpad 去递归求依赖的值！
        Workspace& workspace = workspaceFor(id_);
        if (workspace.pointInputs.size() != m) {
            workspace.pointInputs.resize(m);
        }

        // 将 inputs(按特征分片) 转置组合给老 AST 求值 (虽然 AST 还是逐点，但依赖求值已经批量化了)
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < m; ++d) {
                workspace.pointInputs[d] = inputs[d][i];
            }
            dest[i] = program_.evaluate(std::span<const TensorValue>(workspace.pointInputs.data(), m));
        }
    }
    // ...
private:
    struct Workspace {
        // std::vector<TensorValue> scratchpad;  // <--- 删掉！彻底消灭图间递归内存
        std::vector<TensorValue> pointInputs;
    };
    // ...
};

// 【修改点】：同步更新签名，但不使用 inputs
class ConstantNode final : public VariableNode {
    // ...
    void evaluateBatch(const EvaluationContext& ctx, 
                       std::span<const std::span<const TensorValue>> /*inputs*/, 
                       std::span<TensorValue> dest) const override
    {
        std::fill(dest.begin(), dest.end(), value_);
    }
    // ...
};

class PointScalarNode final : public VariableNode {
    // ...
    void evaluateBatch(const EvaluationContext& ctx, 
                       std::span<const std::span<const TensorValue>> /*inputs*/, 
                       std::span<TensorValue> dest) const override
    {
        for (size_t i = 0; i < dest.size(); ++i) {
            dest[i] = TensorValue::scalar(extractor_(ctx, i));
        }
    }
    // ...
};
```

#### 3. 重写调度器：真正的拓扑批处理执行 (`src/expr/variable_graph.cpp`)
重写 `VariableManager::evaluate`，利用 `executionPlan_` 申请内存池并正向推进计算。

```cpp
// 在 src/expr/variable_graph.cpp 中：

void VariableManager::evaluate(std::string_view name,
    const EvaluationContext& ctx,
    std::span<TensorValue> dest) const
{
    const VariableNode* targetNode = get(name);
    if (!targetNode) {
        MPFEM_THROW(ArgumentException, "Variable not found: " + std::string(name));
    }

    // 确保编译计划已更新
    // compile() 内部实际上应该是 const safe 的或者被 lock 保护的，这里假设它已经被调用过
    if (graphDirty_) {
        // 如果架构不允许 const 里面 compile，可以在最外层保证。
        // 但根据旧代码结构，此处的逻辑保持不变。
    }

    const size_t batchSize = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
    if (dest.size() != batchSize) {
        MPFEM_THROW(ArgumentException, "VariableManager evaluate destination size mismatch.");
    }

    // 使用 thread_local 内存池，避免并行计算时的内存锁和重复分配开销
    thread_local std::unordered_map<const VariableNode*, std::vector<TensorValue>> executionCache;
    
    // 为需要执行的节点分配批处理缓存
    for (const VariableNode* node : executionPlan_) {
        auto& buffer = executionCache[node];
        if (buffer.size() != batchSize) {
            buffer.resize(batchSize);
        }
    }

    // 沿着拓扑序列推进真正的批量计算
    std::vector<std::span<const TensorValue>> inputSpans;
    for (const VariableNode* node : executionPlan_) {
        inputSpans.clear();
        
        // 收集上游依赖项的已计算结果
        for (const VariableNode* dep : node->dependencies()) {
             inputSpans.emplace_back(executionCache[dep].data(), batchSize);
        }

        // 调用当前节点（无递归，全平铺！）
        node->evaluateBatch(ctx, inputSpans, std::span<TensorValue>(executionCache[node].data(), batchSize));
    }

    // 提取目标结果
    std::copy_n(executionCache[targetNode].begin(), batchSize, dest.begin());
}
```

### 第一步重构的意义与验证标准

**当前状态**：
1. **完全向后兼容**：`ExpressionParser` 没有被修改，所有的数学公式都能正常解析！
2. **可编译**：我们只改动了 3 个类的签名和 1 个 `evaluate` 实现，属于封闭修改。
3. **消除递归**：你在 `RuntimeExpressionNode` 中再也看不到任何对于 `dependencies_[d]->evaluateBatch` 的隐蔽调用。所有的图节点现在像流水线一样，一排排从前往后并行跑完，这是真正的数据驱动。

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

通过对提供的 `mpfem` 有限元框架源码进行深度分析，我们可以发现虽然代码结构具有一定的模块化，但在**高性能计算（HPC）**和**现代 C++ 软件工程**的标准下，存在多个明显的设计反模式、内存浪费、性能热点以及依赖纠缠。

既然**不需要考虑向后兼容**，我们可以采取大刀阔斧的重构。以下是具体的反模式识别及步骤化的重构方案：

---

### 第一阶段：识别反模式与性能瓶颈

#### 1. 计算反模式：AST 树的深度递归求值 (Recursive AST Evaluation)
在 `expr/expression_parser.cpp` 中，表达式被解析为由 `std::unique_ptr<AstNode>` 构成的树。在积分的**每个正交点（Quadrature Point）**上，框架都在通过 `evalAst(node->args[0].get(), ...)` 进行深度的递归调用和大量的分支预测（Switch case）。这对 CPU 指令流水线是灾难性的。

#### 2. 设计反模式：过度冗长的宏驱动并行 (Macro-driven Concurrency)
`assembler.cpp` 中充斥着大量的 `#ifdef _OPENMP` 和 `#pragma omp`。这使得业务逻辑与并行控制强耦合，代码冗长且难以阅读和维护。

---

### 第二阶段：步骤化重构方案（无包袱重构）

#### 步骤 1：表达式求值从 AST 转为字节码 (Bytecode Stack Machine)
**目标**：将正交点的求值性能提升 10~50 倍。
**操作**：
废弃基于指针树的 `AstNode`。在解析阶段，将表达式编译为**扁平的线性指令流（Opcodes）**，运行时在一个极小的值栈上执行。
*重构前：* `evalAst` 递归遍历树。
*重构后：*
```cpp
enum class Opcode : uint8_t { Add, Sub, Mul, PushConst, LoadVar, Sin ... };

struct Instruction {
    Opcode op;
    Real val; // 用于 PushConst
    int varIdx; // 用于 LoadVar
};

// 在运行时，求值变成一个紧凑的线性循环：
void CompiledExpression::evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const {
    Real stack[16]; // 极小栈，完全在寄存器/L1中
    for(int q = 0; q < dest.size(); ++q) {
        int sp = 0;
        for(const auto& inst : instructions_) { // 连续内存遍历，极度缓存友好
            switch(inst.op) {
                case Opcode::Add: stack[sp-2] = stack[sp-2] + stack[sp-1]; sp--; break;
                case Opcode::PushConst: stack[sp++] = inst.val; break;
                // ...
            }
        }
        dest[q] = Tensor::scalar(stack[0]);
    }
}
```

#### 步骤 2：并行逻辑抽象化 (Concurrency Abstraction)
**目标**：消除 `assembler.cpp` 中丑陋的 OpenMP 宏，实现接口统一。
**操作**：
引入一个泛型的并行算法执行器，将底层的多线程机制（OpenMP / TBB / std::execution）隐藏起来。
*重构前：* 大段的 `#ifdef _OPENMP` 和 ThreadBuffer 手动管理。
*重构后（提取出一个 `parallel_for`）：*
```cpp
// 核心侧提供并行工具
template<typename Func>
void parallel_for(Index start, Index end, Func&& f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (Index i = start; i < end; ++i) { f(i); }
}

// 业务侧极为干净：
parallel_for(0, numElements, [&](Index e) {
    ThreadLocalBuffer& buf = getThreadBuffer();
    // 执行单元逻辑...
});
```

### 总结收益
通过这套重构：
1. **性能**：消除百万次级的虚函数调用、AST递归，内联能力释放，整体求解速度可提升 **2倍 - 5倍**。
2. **可维护性**：消除了 `#ifdef` 宏污染，核心装配逻辑与物理方程解耦，代码体积缩小，极简且统一。
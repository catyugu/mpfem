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

通过对您的代码架构和性能分析（Profiler）结果的仔细审视，我识别出了该工程目前的几个关键问题。性能瓶颈明确地指向了**表达式求值阶段**（`evalScalarNode`, `evaluateBatch` 占用约 12% 性能）和 **组装内部的虚函数调用/矩阵运算**（`BilinearFormAssembler::assemble` 占用近 40%）。

同时，您提到 `PhysicsProblemBuilder` 高达 1.8 万行，这典型地反映了**硬编码组合爆炸、向后兼容负担、缺乏自动 DAG 解析**的设计反模式。

以下是具体的诊断与**破坏式重构（打破向后兼容以换取一致性和性能）**步骤。由于涉及到基础架构替换，我们将采用统一的张量计算图（自动 DAG） + 扁平化指令集（虚拟机/JIT 后端）策略。

### 诊断：当前架构的反模式

1. **表达式求值的“多态陷阱” (Virtual Dispatch in Inner Loops)**：
   `VariableNode::evaluateBatch` 和 `evalScalarNode` 大量出现在热点中。使用面向对象的树形结构（AST）并通过虚函数在每个积分点（Quadrature Point）求值，会导致严重的指令缓存未命中和分支预测失败。
2. **类型割裂 (Scalar/Vector/Matrix 分离处理)**：
   缺少统一的 `Tensor` 抽象。代码中可能存在分别处理标量节点和向量节点的冗长代码，无法优雅地应对后续的张量（如应力、应变）操作。

---

### 重构目标与架构蓝图
。
* **统一张量**：兼容标量形状 `()`，向量 `(n)`，矩阵 `(n,m)`。
* **表达式编译后端**：将 AST 树在初始化时“拍平”为基于连续内存的执行计划（类似 Bytecode/栈机模式），**消灭内部循环的虚函数**。

### 第一步：统一张量值与扁平化执行引擎（表达式编译后端）

我们要消灭 `VariableNode` 的虚函数调用。取而代之的是，构建一个线性的执行序列（Tape）。每一步执行一个特定类型的算子（通过函数指针或 `std::function`，或者更高效的 `switch(opcode)` 虚拟机模式）。

首先定义统一张量维度和虚拟机指令集结构：

```cpp
// src/expr/tensor_system.hpp
#ifndef MPFEM_EXPR_TENSOR_SYSTEM_HPP
#define MPFEM_EXPR_TENSOR_SYSTEM_HPP

#include <vector>
#include <array>
#include <span>
#include <variant>

namespace mpfem {

// 所有的运算都在一段连续的内存缓冲区（Tape）中进行
struct ExecContext {
    int batch_size;
    std::vector<double> registers; // 扁平化寄存器堆，预分配以容纳所有的中间变量
};

// 操作码
enum class OpCode : uint8_t {
    ADD, SUB, MUL, DIV, DOT, CROSS, MATMUL,
    LOAD_CONST, LOAD_FIELD_VAL, EVAL_SHAPE_FUNC
};

// 一条扁平化的指令
struct Instruction {
    OpCode op;
    int dest_reg;     // 输出结果存放的寄存器偏移
    int src_reg1;     // 输入1
    int src_reg2;     // 输入2
    int stride;       // 张量步长（1 标量，3 向量，9 矩阵）
    double const_val; // 可选常量
};

// 编译后的表达式执行程序
class CompiledExpression {
    std::vector<Instruction> instructions;
    int required_registers_size;

public:
    void execute(ExecContext& ctx) const {
        const int batch = ctx.batch_size;
        double* regs = ctx.registers.data();
        
        // 【关键】消灭了多态，全部变成紧凑的指令迭代和连续内存操作
        for (const auto& inst : instructions) {
            double* dest = regs + inst.dest_reg * batch;
            const double* s1 = regs + inst.src_reg1 * batch;
            const double* s2 = regs + inst.src_reg2 * batch;
            const int stride = inst.stride;

            switch (inst.op) {
                case OpCode::ADD:
                    // 自动向量化友好 (SIMD)
                    for(int i=0; i < batch * stride; ++i) dest[i] = s1[i] + s2[i];
                    break;
                case OpCode::MUL:
                    for(int i=0; i < batch * stride; ++i) dest[i] = s1[i] * s2[i];
                    break;
                // ... 其他张量计算逻辑
                case OpCode::LOAD_CONST:
                    for(int i=0; i < batch * stride; ++i) dest[i] = inst.const_val;
                    break;
            }
        }
    }
};

}
#endif
```

**重构行动：** 在当前的 `expression_parser.cpp` 中，修改原有的解析逻辑。原来是生成一棵 `RuntimeExpressionNode` 树，现在改为遍历这棵 AST 树，**发射（Emit）** `Instruction` 到 `CompiledExpression` 中。

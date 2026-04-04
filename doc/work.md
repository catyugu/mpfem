# 大纲

这是一项非常冗长的任务，建议您充分利用完整的输出上下文，充分利用子agent来处理。
建议您充分利用完整的输出上下文来处理——整体输入和输出 tokens 控制在 200k tokens，充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。充分利用上下文窗口长度将任务彻底完成，避免耗尽 tokens。

## 原则

* 任何情况下，逻辑嵌套必须少于三层。
* 代码越精简越好，抹除不必要的抽象。
* 尽可能少做判断，只在最接近用户层的地方做判断，减少热循环中分支预测代价。
* 所有同质功能的接口只保留一个性能最高、最易用的，使代码更清晰，不易误用。
* 禁止使用const_cast（除非为了调用外部求解器的局部使用），mutable（除非为了缓存或者锁），friend，dynamic_cast，shared_ptr等关键字或功能。
* 删除冗余的成员变量、接口等。
* 尽量使用pimpl模式最小化编译依赖与交叉耦合。
* 把工作任务分成多个子任务，从最容易的子任务开始，完成一块子任务后：
  * 确保编译通过。
  * 确保回归测试通过。
  * 验证`doc/validation.md`全部案例。
  * 拒绝向后兼容性，强制改写所有调用处，让代码更简洁，对以后的扩展更通用。
  * 验证编译运行结果，移除所有向后兼容的或容易误用的接口，防止冗余。
  * 提交一次代码，然后继续完成下一个子任务。

## 具体工作任务

### 任务一：重塑统一算子抽象 (LinearOperator 基类)

废弃原有的“求解器 vs 预条件子”二元对立设计，建立“一切皆算子”的统一抽象模型。无论是直接法、Krylov 方法、AMG 还是 DDM，均必须继承自统一的基类接口。

* **定义核心两阶段生命周期：**
    * 强制实现 `setup(const Matrix* A)`：处理矩阵因式分解、区域切分、粗网格构建等繁重计算。
    * 强制实现 `apply(const Vector& b, Vector& x)`：执行纯粹的向量迭代与映射操作。
* **实现嵌套能力：** 在基类中提供 `set_preconditioner(std::unique_ptr<LinearOperator> pc)` 方法，支持算子的无限层级嵌套。
* **剥离矩阵所有权：** 确保基类及各子类在 `setup` 中绝不隐式接管或销毁外部传入的系统矩阵。

### 任务二：构建动态配置树与工厂解析器

彻底改变线性求解层的实例化方式，从代码硬编码转向基于配置文件的树状解析。

* **实现递归工厂模式：** 编写能够解析多层级 XML 配置的 Factory 方法，依据配置动态实例化算子，并利用 `set_preconditioner` 自动完成组装。
* **支持标准配置协议：** 解析器必须完美支持以下洋葱状的嵌套 XML 结构：

```xml
<SolverConfiguration>
    <Operator type="CG">
        <Parameters>
            <Tolerance>1e-10</Tolerance>
            <MaxIterations>1000</MaxIterations>
            <PrintLevel>0</PrintLevel>
        </Parameters>
        <Preconditioner>
            <Operator type="Jacobi">
                <Parameters>
                    <Sweeps>1</Sweeps>
                </Parameters>
            </Operator>
        </Preconditioner>
    </Operator>
</SolverConfiguration>
```

### 任务三：解决动静多态冲突 (Eigen Adapter)

针对底层大量使用 Eigen 等基于模板元编程的静态求解器，开发适配器以桥接动态架构。

* **编写 Adapter 封装类：** 创建符合 Eigen 预条件子 Concept 的模板类 `DynamicPreconditionerAdapter`。
* **运行时转发机制：** 该 Adapter 内部持有 `LinearOperator*` 借用指针，在 Eigen 的编译期循环内，通过虚函数调用将 `solve` 操作安全转发至动态配置的自定义算子中。

### 任务四：全局重构与测试验证

* **清理旧架构：** 完全移除原有的 Solver 层遗留代码。
* **升级测试用例：** 修改项目中所有 `case.xml` 文件，使其符合全新的嵌套算子配置规范。
* **端到端验证：** 执行系统级编译，并运行全套测试集，确保精度和性能符合预期，且借助内存检测工具（如 Valgrind/ASan）确认无由 `unique_ptr` 和裸指针混用导致的内存泄漏或悬空指针。
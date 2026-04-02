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

## 工作任务1

* 彻底改变线性求解层的架构设计，允许多层次的配置/求解方式，参考配置格式：

```xml
<SolverConfiguration>
    <LinearSolver type="DGMRES">
        <Parameters>
            <Tolerance>1e-8</Tolerance>
            <MaxIterations>500</MaxIterations>
            <Restart>30</Restart>
        </Parameters>

        <Preconditioner type="AdditiveSchwarz">
            <Parameters>
                <Overlap>1</Overlap> </Parameters>

            <LocalSolver type="ILU">
                <Parameters>
                    <FillLevel>0</FillLevel>
                    <DropTolerance>1e-4</DropTolerance>
                </Parameters>
            </LocalSolver>

            <CoarseSolver type="AMG">
                <Parameters>
                    <CoarseningMethod>Ruge-Stueben</CoarseningMethod>
                    <CycleType>V-Cycle</CycleType>
                    <MaxLevels>5</MaxLevels>
                </Parameters>
                
                <Smoother type="GaussSeidel">
                    <Parameters>
                        <Sweeps>2</Sweeps>
                    </Parameters>
                </Smoother>
            </CoarseSolver>

        </Preconditioner>
    </LinearSolver>
</SolverConfiguration>
```

* 彻底分离求解器与预条件设置（例如求解器只提供单纯的LU，CG，DGMRES等，预条件另外设置），重写相关工厂和io方法。
* 修改几个case.xml以使用新的模式，编译，测试，验证。

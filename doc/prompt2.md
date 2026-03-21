# 项目：mpfem

本文档是doc/prompt1.md的后续，请先阅读doc/prompt1.md以获知本项目的基本信息和代码规范

## 新的要求

* 接下来我们需要时变（准静态电场，热场，位移场）求解，但由于旧代码特化于静态非线性耦合问题，需要架构上做若干改进。
* 你需要修改案例文件的格式（也许也需要busbar_steady和busbar_steady_order2的case.xml来指明它们是稳态。
* 需要添加对状态向量的统一管理与“算子”的抽象以便一般地对线性、非线性问题进行处理。
* 优先支持分离式迭代，调试完成后再引入完整式迭代。

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

### 一阶测试：`cases/busbar_transient/`下

* 案例描述：一个母线板，上面有三个螺丝。求解稳态问题。
* 网格文件：mesh.mphtxt
* 配置文件：case.xml
* 时间步进要求：0.1秒时间步长，开始时间t=0，结束时间t=1
* 一共7个domain，43个boundary。（到此你应该检验网格读入是否正确）
* 三个物理场：电势（准静态）、固体传热、刚体力学。
* 材料：Copper用于domain 1（板体），Titanium beta-21S用于domain 2-7，材料属性见于material.xml
* 边界条件：
  * 电势：boundary 43电势为Vtot，boundary 8和15电势为0（接地）。其余外边界绝缘。
  * 固体传热：boundary 1-7, 9-14, 16-42有对流换热边界，换热系数为htc，外界温度为293.15K，其余外边界热绝缘。
  * 刚体力学：boundary 8, 15, 43 为固定（零位移）。
* 初始条件：
  * 电势0
  * 温度293.15K
  * 位移0，速度0
* 考虑焦耳热和热应变。
* COMSOL结果文件：result.txt。
* 请将我们的结果和COMSOL结果文件进行逐个时间步对比。（这一步可以用Python等外部脚本完成，C++代码只需要能export标准格式的结果即可）
* 先使用一阶BDF（后向欧拉）方法，跑通之后再做BDF2，CrankNicolson。
* 要求所有时间步电势场相对L2误差<1e-5，温度场相对L2误差<1e-7，位移场相对误差<1e-3

### 二阶测试：`cases/busbar_steady_order2/`下

* 其余与一阶情况一样，只是网格为二阶，基函数也为二阶多项式
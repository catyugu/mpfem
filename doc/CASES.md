# 案例说明

## busbar一阶稳态电热力耦合算例：`cases/busbar_steady/`

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

## busbar二阶稳态电热力耦合算例：`cases/busbar_steady_order2/`

* 其余与一阶情况一样，只是网格为二阶，基函数也为二阶多项式

## busbar一阶瞬态电热力耦合算例：`cases/busbar_transient/`

* 案例描述：一个母线板，上面有三个螺丝。求解稳态问题。
* 网格文件：mesh.mphtxt
* 配置文件：case.xml
* 时间步进要求：时间步长10s，开始时间t=0，结束时间t=100s，共11个点。
* 一共7个domain，43个boundary。（到此你应该检验网格读入是否正确）
* 三个物理场：电势（准静态，但是电导率有温变）、固体传热、刚体力学。
* 材料：Copper用于domain 1（板体），Titanium beta-21S用于domain 2-7，材料属性见于material.xml
* 边界条件：
  * 电势：boundary 43电势为Vtot，boundary 8和15电势为0（接地）。其余外边界绝缘。
  * 固体传热：boundary 1-7, 9-14, 16-42有对流换热边界，换热系数为htc，外界温度为293.15K，其余外边界热绝缘。
  * 刚体力学：boundary 8, 15, 43 为固定（零位移）。
* 初始条件：
  * 域1-7，电势0
  * 域1-7，温度293.15K
  * 域1-7，位移0，速度0
* 考虑焦耳热和热应变。
* COMSOL结果文件：result.txt。
* 结果输出形式：按时间片导出为vtu/全部结果导出成和COMSOL相同格式的txt
* 请将我们的结果和COMSOL结果文件进行逐个时间步对比。（这一步可以用Python等外部脚本完成，C++代码只需要能输出正确格式的结果即可）
* 先使用一阶BDF（后向欧拉）方法，跑通之后再做BDF2，CrankNicolson。
* 要求所有时间步电势场相对L2误差<1e-5，温度场相对L2误差<1e-4，位移场相对误差<1e-2
# 验证准则

## 1阶稳态

### 运行方式

`.\build-llvm\examples\busbar_example.exe .\cases\busbar_steady`
`python .\scripts\compare_steady_results.py .\results\mpfem_result.txt .\cases\busbar_steady\result.txt`

### 参考结果

```
field   L2      Linf    max_relative    L2_relative
V       1.398699e-08    2.095434e-08    2.811702e-06    1.912433e-06
T       2.553170e-05    2.817855e-05    8.711728e-08    7.883343e-08
disp    2.420344e-11    4.662133e-11    9.686668e-07    8.734340e-07
```

## 2阶稳态

`.\build-llvm\examples\busbar_example.exe .\cases\busbar_steady_order2`
`python .\scripts\compare_steady_results.py .\results\mpfem_result.txt .\cases\busbar_steady_order2\result.txt`

### 参考结果

```
V       1.236373e-08    1.847981e-08    2.489623e-06    1.694028e-06
T       1.924976e-05    2.147747e-05    6.665505e-08    5.966417e-08
disp    5.568379e-09    1.841973e-08    5.390639e-03    2.097754e-04
```

## 1阶瞬态

`./build-llvm/examples/busbar_example .\cases\busbar_transient`
`python .\scripts\compare_transient_results.py .\results\mpfem_result.txt .\cases\busbar_transient\result.txt`

### 参考结果

```
Time Step       V L2            V L2 Rel        T L2            T L2 Rel        Disp L2         Disp L2 Rel     Status
----------------------------------------------------------------------------------------------------
t=0             1.46e-12                2.01e-10                3.64e-05                1.24e-07                1.16e-11           0.00e+00         PASS
t=10            2.38e-08                3.28e-06                8.48e-02                2.89e-04                2.08e-08           1.02e-01         FAIL
t=20            1.98e-08                2.73e-06                5.54e-02                1.89e-04                1.32e-08           3.71e-02         FAIL
t=30            1.01e-08                1.39e-06                2.39e-02                8.14e-05                5.77e-09           1.15e-02         FAIL
t=40            4.82e-09                6.62e-07                1.05e-02                3.58e-05                2.62e-09           4.01e-03         PASS
t=50            5.76e-10                7.92e-08                4.85e-03                1.65e-05                1.47e-09           1.81e-03         PASS
t=60            7.47e-10                1.03e-07                2.46e-03                8.36e-06                7.87e-10           8.10e-04         PASS
t=70            6.88e-10                9.45e-08                1.50e-03                5.08e-06                5.57e-10           4.92e-04         PASS
t=80            7.95e-10                1.09e-07                1.13e-03                3.82e-06                5.65e-10           4.36e-04         PASS
t=90            1.05e-09                1.45e-07                1.07e-03                3.61e-06                6.67e-10           4.56e-04         PASS
t=100           2.27e-09                3.12e-07                1.51e-03                5.12e-06                9.81e-10           6.04e-04         PASS
----------------------------------------------------------------------------------------------------

```
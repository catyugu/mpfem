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
t=10            9.22e-09                1.27e-06                3.02e-02             1.03e-04         7.33e-09                3.25e-02                FAIL
t=20            2.53e-09                3.48e-07                6.57e-03             2.24e-05         1.58e-09                4.37e-03                PASS
t=30            1.14e-09                1.57e-07                2.90e-03             9.87e-06         7.32e-10                1.46e-03                PASS
t=40            2.17e-10                2.98e-08                1.33e-03             4.53e-06         3.57e-10                5.45e-04                PASS
t=50            2.51e-09                3.45e-07                6.95e-04             2.36e-06         4.01e-10                4.95e-04                PASS
t=60            1.66e-09                2.28e-07                4.53e-04             1.54e-06         2.59e-10                2.67e-04                PASS
t=70            9.04e-10                1.24e-07                3.80e-04             1.29e-06         2.44e-10                2.15e-04                PASS
t=80            6.15e-10                8.45e-08                3.73e-04             1.26e-06         2.88e-10                2.22e-04                PASS
t=90            5.77e-10                7.94e-08                4.65e-04             1.58e-06         3.75e-10                2.57e-04                PASS
t=100           1.58e-09                2.17e-07                9.76e-04             3.31e-06         6.70e-10                4.12e-04                PASS
----------------------------------------------------------------------------------------------------

```
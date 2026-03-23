# 验证准则

## 1阶稳态

`.\build-llvm\examples\busbar_example.exe .\cases\busbar_steady`
`python .\scripts\compare_steady_results.py .\results\mpfem_result.txt .\cases\busbar_steady\result.txt`

## 2阶稳态

`.\build-llvm\examples\busbar_example.exe .\cases\busbar_steady_order2`
`python .\scripts\compare_steady_results.py .\results\mpfem_result.txt .\cases\busbar_steady_order2\result.txt`

## 1阶瞬态

`./build-llvm/examples/busbar_example .\cases\busbar_transient`
`python .\scripts\compare_transient_results.py .\results\mpfem_result.txt .\cases\busbar_transient\result.txt`
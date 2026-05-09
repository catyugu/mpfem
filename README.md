# MPFEM: 高性能的电、热、力耦合有限元仿真内核

## 支持功能

- 表达式解析
- 稳态/瞬态分析
- vtu格式/comsol结果格式输出

## Windows平台使用独立环境的编译方式

```cmd
# 使用conda虚拟环境，需已安装clang, cmake等
conda activate numerical

cmake -S . -B build-clang -DCMAKE_BUILD_TYPE=Release
cmake --build build-clang --parallel

# 运行示例
build-clang/examples/busbar_example.exe ./cases/busbar_steady_order2
```
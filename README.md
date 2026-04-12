# MPFEM: 高性能的电、热、力耦合有限元仿真内核

## 支持功能

- 表达式解析
- 稳态/瞬态分析
- vtu格式/comsol结果格式输出

## 编译方式

### 使用MSVC风格编译器

```bash
E:\env\cpp\VS14\Common7\Tools\VsDevCmd.bat
cmake -S . -B build-msvc
cmake --build build-msvc --parallel --config=Release
# 运行示例
build-llvm\examples\busbar_example.exe
```

### 使用LLVM风格编译器

```bash
E:\env\cpp\msys2\msys2_shell.cmd -defterm -here -no-start -clang64
cmake -S . -B build-llvm
cmake --build build-llvm --parallel
# 运行示例
build-msvc\examples\Release\busbar_example.exe
```
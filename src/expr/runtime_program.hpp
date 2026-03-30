#ifndef MPFEM_EXPR_RUNTIME_PROGRAM_HPP
#define MPFEM_EXPR_RUNTIME_PROGRAM_HPP

#include "expr/expression_parser.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mpfem {

using RuntimeSymbolResolver = std::function<bool(std::string_view, double&)>;

struct RuntimeSymbolConfig {
    std::vector<std::pair<std::string, double>> constants;
    std::vector<std::string> dynamicSymbols;
};

class CompiledScalarRuntimeProgram {
public:
    CompiledScalarRuntimeProgram(std::string expression, RuntimeSymbolConfig config);
    ~CompiledScalarRuntimeProgram();
    CompiledScalarRuntimeProgram(CompiledScalarRuntimeProgram&&) noexcept;
    CompiledScalarRuntimeProgram& operator=(CompiledScalarRuntimeProgram&&) noexcept;
    CompiledScalarRuntimeProgram(const CompiledScalarRuntimeProgram&) = delete;
    CompiledScalarRuntimeProgram& operator=(const CompiledScalarRuntimeProgram&) = delete;

    double evaluate(const RuntimeSymbolResolver& resolver) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class CompiledMatrixRuntimeProgram {
public:
    CompiledMatrixRuntimeProgram(std::string expression, RuntimeSymbolConfig config);
    ~CompiledMatrixRuntimeProgram();
    CompiledMatrixRuntimeProgram(CompiledMatrixRuntimeProgram&&) noexcept;
    CompiledMatrixRuntimeProgram& operator=(CompiledMatrixRuntimeProgram&&) noexcept;
    CompiledMatrixRuntimeProgram(const CompiledMatrixRuntimeProgram&) = delete;
    CompiledMatrixRuntimeProgram& operator=(const CompiledMatrixRuntimeProgram&) = delete;

    Matrix3 evaluate(const RuntimeSymbolResolver& resolver) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mpfem

#endif  // MPFEM_EXPR_RUNTIME_PROGRAM_HPP

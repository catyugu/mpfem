#include "expr/runtime_program.hpp"

#include "core/exception.hpp"

#include <atomic>
#include <unordered_map>
#include <utility>

namespace mpfem {
namespace {

class RuntimeVariableStorage {
public:
    explicit RuntimeVariableStorage(const RuntimeSymbolConfig& config) {
        values_.reserve(config.constants.size() + config.dynamicSymbols.size());
        dynamicIndices_.reserve(config.dynamicSymbols.size());
        dynamicSymbols_.reserve(config.dynamicSymbols.size());

        for (const auto& [name, value] : config.constants) {
            addSymbol(name, value);
        }

        for (const std::string& name : config.dynamicSymbols) {
            const size_t index = addSymbol(name, 0.0);
            dynamicIndices_.push_back(index);
            dynamicSymbols_.push_back(name);
        }

        bindings_.reserve(values_.size());
        for (NamedValue& value : values_) {
            bindings_.push_back(ExpressionParser::VariableBinding{value.name, &value.value});
        }
    }

    void setDynamic(size_t index, double value) {
        MPFEM_ASSERT(index < dynamicIndices_.size(), "Dynamic symbol index is out of range.");
        values_[dynamicIndices_[index]].value = value;
    }

    const std::string& dynamicSymbol(size_t index) const {
        MPFEM_ASSERT(index < dynamicSymbols_.size(), "Dynamic symbol index is out of range.");
        return dynamicSymbols_[index];
    }

    size_t dynamicCount() const {
        return dynamicSymbols_.size();
    }

    const std::vector<ExpressionParser::VariableBinding>& bindings() const {
        return bindings_;
    }

private:
    struct NamedValue {
        std::string name;
        double value = 0.0;
    };

    size_t addSymbol(const std::string& name, double value) {
        for (size_t i = 0; i < values_.size(); ++i) {
            if (values_[i].name == name) {
                MPFEM_THROW(ArgumentException,
                            "Duplicated runtime symbol configuration for: " + name);
            }
        }

        values_.push_back(NamedValue{name, value});
        return values_.size() - 1;
    }

    std::vector<NamedValue> values_;
    std::vector<size_t> dynamicIndices_;
    std::vector<std::string> dynamicSymbols_;
    std::vector<ExpressionParser::VariableBinding> bindings_;
};

uint64_t nextProgramId() {
    static std::atomic<uint64_t> id{1};
    return id.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

struct CompiledScalarRuntimeProgram::Impl {
    struct ThreadState {
        RuntimeVariableStorage variables;
        ExpressionParser::ScalarProgram program;

        ThreadState(const std::string& expression, const RuntimeSymbolConfig& config)
            : variables(config),
              program(ExpressionParser::instance().compileScalar(expression, variables.bindings())) {
        }
    };

    explicit Impl(std::string expressionText, RuntimeSymbolConfig symbolConfig)
        : expression(std::move(expressionText)),
          config(std::move(symbolConfig)),
          id(nextProgramId()) {
    }

    double evaluate(const RuntimeSymbolResolver& resolver) const {
        ThreadState& state = getThreadState();

        for (size_t i = 0; i < state.variables.dynamicCount(); ++i) {
            double value = 0.0;
            const std::string& symbol = state.variables.dynamicSymbol(i);
            if (!resolver || !resolver(symbol, value)) {
                MPFEM_THROW(ArgumentException,
                            "Runtime symbol resolver failed for symbol: " + symbol);
            }
            state.variables.setDynamic(i, value);
        }

        return state.program.evaluate();
    }

    ThreadState& getThreadState() const {
        static thread_local std::unordered_map<uint64_t, ThreadState> cache;

        auto it = cache.find(id);
        if (it != cache.end()) {
            return it->second;
        }

        auto inserted = cache.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id),
            std::forward_as_tuple(expression, config));
        return inserted.first->second;
    }

    std::string expression;
    RuntimeSymbolConfig config;
    uint64_t id;
};

struct CompiledMatrixRuntimeProgram::Impl {
    struct ThreadState {
        RuntimeVariableStorage variables;
        ExpressionParser::MatrixProgram program;

        ThreadState(const std::string& expression, const RuntimeSymbolConfig& config)
            : variables(config),
              program(ExpressionParser::instance().compileMatrix(expression, variables.bindings())) {
        }
    };

    explicit Impl(std::string expressionText, RuntimeSymbolConfig symbolConfig)
        : expression(std::move(expressionText)),
          config(std::move(symbolConfig)),
          id(nextProgramId()) {
    }

    Matrix3 evaluate(const RuntimeSymbolResolver& resolver) const {
        ThreadState& state = getThreadState();

        for (size_t i = 0; i < state.variables.dynamicCount(); ++i) {
            double value = 0.0;
            const std::string& symbol = state.variables.dynamicSymbol(i);
            if (!resolver || !resolver(symbol, value)) {
                MPFEM_THROW(ArgumentException,
                            "Runtime symbol resolver failed for symbol: " + symbol);
            }
            state.variables.setDynamic(i, value);
        }

        return state.program.evaluate();
    }

    ThreadState& getThreadState() const {
        static thread_local std::unordered_map<uint64_t, ThreadState> cache;

        auto it = cache.find(id);
        if (it != cache.end()) {
            return it->second;
        }

        auto inserted = cache.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id),
            std::forward_as_tuple(expression, config));
        return inserted.first->second;
    }

    std::string expression;
    RuntimeSymbolConfig config;
    uint64_t id;
};

CompiledScalarRuntimeProgram::CompiledScalarRuntimeProgram(std::string expression, RuntimeSymbolConfig config)
    : impl_(std::make_unique<Impl>(std::move(expression), std::move(config))) {
}

CompiledScalarRuntimeProgram::~CompiledScalarRuntimeProgram() = default;
CompiledScalarRuntimeProgram::CompiledScalarRuntimeProgram(CompiledScalarRuntimeProgram&&) noexcept = default;
CompiledScalarRuntimeProgram& CompiledScalarRuntimeProgram::operator=(CompiledScalarRuntimeProgram&&) noexcept = default;

double CompiledScalarRuntimeProgram::evaluate(const RuntimeSymbolResolver& resolver) const {
    return impl_->evaluate(resolver);
}

CompiledMatrixRuntimeProgram::CompiledMatrixRuntimeProgram(std::string expression, RuntimeSymbolConfig config)
    : impl_(std::make_unique<Impl>(std::move(expression), std::move(config))) {
}

CompiledMatrixRuntimeProgram::~CompiledMatrixRuntimeProgram() = default;
CompiledMatrixRuntimeProgram::CompiledMatrixRuntimeProgram(CompiledMatrixRuntimeProgram&&) noexcept = default;
CompiledMatrixRuntimeProgram& CompiledMatrixRuntimeProgram::operator=(CompiledMatrixRuntimeProgram&&) noexcept = default;

Matrix3 CompiledMatrixRuntimeProgram::evaluate(const RuntimeSymbolResolver& resolver) const {
    return impl_->evaluate(resolver);
}

}  // namespace mpfem

#include "io/compiled_expression_coefficient.hpp"

#include "core/exception.hpp"
#include "fe/element_transform.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "io/expression_symbol_usage.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mpfem {
namespace {

class RuntimeExpressionContext {
public:
    RuntimeExpressionContext(const CaseDefinition& caseDef, const ExpressionSymbolUsage& usage) {
        const size_t runtimeSymbolCount = static_cast<size_t>(usage.useTime) +
                                          (usage.useSpace ? 3u : 0u) +
                                          static_cast<size_t>(usage.useTemperature) +
                                          static_cast<size_t>(usage.usePotential);
        values_.reserve(usage.caseVariables.size() + runtimeSymbolCount);

        for (const std::string& name : usage.caseVariables) {
            const auto it = caseDef.variableMap_.find(name);
            if (it != caseDef.variableMap_.end()) {
                addSymbol(it->first, it->second);
            }
        }

        if (usage.useTime) {
            t_ = addSymbol("t", 0.0);
        }
        if (usage.useSpace) {
            x_ = addSymbol("x", 0.0);
            y_ = addSymbol("y", 0.0);
            z_ = addSymbol("z", 0.0);
        }
        if (usage.useTemperature) {
            temperature_ = addSymbol("T", 293.15);
        }
        if (usage.usePotential) {
            potential_ = addSymbol("V", 0.0);
        }

        bindings_.reserve(values_.size());
        for (NamedValue& entry : values_) {
            bindings_.push_back(ExpressionParser::VariableBinding{entry.name, &entry.value});
        }
    }

    void updateTime(Real time) {
        if (t_) {
            *t_ = time;
        }
    }

    void updateSpace(ElementTransform& trans) {
        if (!x_ || !y_ || !z_) {
            return;
        }

        Vector3 position;
        trans.transform(trans.integrationPoint(), position);
        *x_ = position.x();
        *y_ = position.y();
        *z_ = position.z();
    }

    void updateTemperature(Real value) {
        if (temperature_) {
            *temperature_ = value;
        }
    }

    void updatePotential(Real value) {
        if (potential_) {
            *potential_ = value;
        }
    }

    const std::vector<ExpressionParser::VariableBinding>& bindings() const {
        return bindings_;
    }

private:
    struct NamedValue {
        std::string name;
        double value = 0.0;
    };

    double* addSymbol(std::string name, double value) {
        for (NamedValue& entry : values_) {
            if (entry.name == name) {
                return &entry.value;
            }
        }

        values_.push_back(NamedValue{std::move(name), value});
        return &values_.back().value;
    }

    std::vector<NamedValue> values_;
    std::vector<ExpressionParser::VariableBinding> bindings_;
    double* t_ = nullptr;
    double* x_ = nullptr;
    double* y_ = nullptr;
    double* z_ = nullptr;
    double* temperature_ = nullptr;
    double* potential_ = nullptr;
};

void updateRuntimeContext(ElementTransform& trans,
                          Real t,
                          const ExpressionSymbolUsage& usage,
                          const ExpressionFieldAccessors& fieldAccessors,
                          RuntimeExpressionContext& context) {
    if (usage.useTime) {
        context.updateTime(t);
    }
    if (usage.useSpace) {
        context.updateSpace(trans);
    }
    if (usage.useTemperature) {
        Real value = 0.0;
        const bool ok = fieldAccessors.sampleTemperature && fieldAccessors.sampleTemperature(trans, t, value);
        if (!ok) {
            MPFEM_THROW(ArgumentException,
                        "Expression requires symbol 'T' but heat-transfer field is unavailable at runtime.");
        }
        context.updateTemperature(value);
    }
    if (usage.usePotential) {
        Real value = 0.0;
        const bool ok = fieldAccessors.samplePotential && fieldAccessors.samplePotential(trans, t, value);
        if (!ok) {
            MPFEM_THROW(ArgumentException,
                        "Expression requires symbol 'V' but electrostatics field is unavailable at runtime.");
        }
        context.updatePotential(value);
    }
}

void validateFieldAccessors(const ExpressionSymbolUsage& usage,
                            const ExpressionFieldAccessors& fieldAccessors) {
    if (usage.useTemperature && !fieldAccessors.sampleTemperature) {
        MPFEM_THROW(ArgumentException,
                    "Expression requires symbol 'T' but no temperature accessor was provided.");
    }
    if (usage.usePotential && !fieldAccessors.samplePotential) {
        MPFEM_THROW(ArgumentException,
                    "Expression requires symbol 'V' but no potential accessor was provided.");
    }
}

class CompiledScalarExpressionCoefficient final : public Coefficient {
public:
    CompiledScalarExpressionCoefficient(std::string expression,
                                        const CaseDefinition* caseDef,
                                        ExpressionSymbolUsage usage,
                                        ExpressionFieldAccessors fieldAccessors)
        : id_(nextId()),
          expression_(std::move(expression)),
          caseDef_(caseDef),
          usage_(std::move(usage)),
          fieldAccessors_(std::move(fieldAccessors)) {
        MPFEM_ASSERT(caseDef_ != nullptr, "Case definition is required for expression coefficient.");
        validateFieldAccessors(usage_, fieldAccessors_);
    }

    void eval(ElementTransform& trans, Real& result, Real t = 0.0) const override {
        ThreadState& state = getThreadState();
        updateRuntimeContext(trans, t, usage_, fieldAccessors_, *state.context);
        result = state.program.evaluate();
    }

private:
    struct ThreadState {
        ExpressionParser::ScalarProgram program;
        std::unique_ptr<RuntimeExpressionContext> context;
    };

    static uint64_t nextId() {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    ThreadState& getThreadState() const {
        static thread_local std::unordered_map<uint64_t, ThreadState> cache;

        auto it = cache.find(id_);
        if (it != cache.end()) {
            return it->second;
        }

        auto context = std::make_unique<RuntimeExpressionContext>(*caseDef_, usage_);
        auto program = ExpressionParser::instance().compileScalar(expression_, context->bindings());

        ThreadState state;
        state.program = std::move(program);
        state.context = std::move(context);

        auto inserted = cache.emplace(id_, std::move(state));
        return inserted.first->second;
    }

    uint64_t id_;
    std::string expression_;
    const CaseDefinition* caseDef_;
    ExpressionSymbolUsage usage_;
    ExpressionFieldAccessors fieldAccessors_;
};

class CompiledMatrixExpressionCoefficient final : public MatrixCoefficient {
public:
    CompiledMatrixExpressionCoefficient(std::string expression,
                                        const CaseDefinition* caseDef,
                                        ExpressionSymbolUsage usage,
                                        ExpressionFieldAccessors fieldAccessors)
        : id_(nextId()),
          expression_(std::move(expression)),
          caseDef_(caseDef),
          usage_(std::move(usage)),
          fieldAccessors_(std::move(fieldAccessors)) {
        MPFEM_ASSERT(caseDef_ != nullptr, "Case definition is required for expression coefficient.");
        validateFieldAccessors(usage_, fieldAccessors_);
    }

    void eval(ElementTransform& trans, Matrix3& result, Real t = 0.0) const override {
        ThreadState& state = getThreadState();
        updateRuntimeContext(trans, t, usage_, fieldAccessors_, *state.context);
        result = state.program.evaluate();
    }

private:
    struct ThreadState {
        ExpressionParser::MatrixProgram program;
        std::unique_ptr<RuntimeExpressionContext> context;
    };

    static uint64_t nextId() {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    ThreadState& getThreadState() const {
        static thread_local std::unordered_map<uint64_t, ThreadState> cache;

        auto it = cache.find(id_);
        if (it != cache.end()) {
            return it->second;
        }

        auto context = std::make_unique<RuntimeExpressionContext>(*caseDef_, usage_);
        auto program = ExpressionParser::instance().compileMatrix(expression_, context->bindings());

        ThreadState state;
        state.program = std::move(program);
        state.context = std::move(context);

        auto inserted = cache.emplace(id_, std::move(state));
        return inserted.first->second;
    }

    uint64_t id_;
    std::string expression_;
    const CaseDefinition* caseDef_;
    ExpressionSymbolUsage usage_;
    ExpressionFieldAccessors fieldAccessors_;
};

}  // namespace

std::unique_ptr<Coefficient> createCompiledScalarExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExpressionFieldAccessors fieldAccessors) {
    const auto usage = analyzeExpressionSymbolUsage(expression, caseDef);
    return std::make_unique<CompiledScalarExpressionCoefficient>(
        std::move(expression),
        &caseDef,
        usage,
        std::move(fieldAccessors));
}

std::unique_ptr<MatrixCoefficient> createCompiledMatrixExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExpressionFieldAccessors fieldAccessors) {
    const auto usage = analyzeExpressionSymbolUsage(expression, caseDef);
    return std::make_unique<CompiledMatrixExpressionCoefficient>(
        std::move(expression),
        &caseDef,
        usage,
        std::move(fieldAccessors));
}

}  // namespace mpfem

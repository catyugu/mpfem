#include "problem/expression_coefficient_factory.hpp"

#include "expr/runtime_program.hpp"
#include "expr/symbol_scanner.hpp"
#include "fe/element_transform.hpp"

#include <utility>

namespace mpfem {
namespace {

RuntimeSymbolConfig buildSymbolConfig(const std::string& expression, const CaseDefinition& caseDef) {
    RuntimeSymbolConfig config;
    const std::vector<std::string> symbols = collectExpressionSymbols(expression);

    config.constants.reserve(symbols.size());
    config.dynamicSymbols.reserve(symbols.size());

    for (const std::string& symbol : symbols) {
        auto it = caseDef.variableMap_.find(symbol);
        if (it != caseDef.variableMap_.end()) {
            config.constants.emplace_back(it->first, it->second);
            continue;
        }
        config.dynamicSymbols.push_back(symbol);
    }

    return config;
}

bool resolveBuiltInSymbol(std::string_view symbol,
                          ElementTransform& transform,
                          Real t,
                          bool& positionCached,
                          Vector3& position,
                          double& value) {
    if (symbol == "t") {
        value = t;
        return true;
    }

    if (symbol != "x" && symbol != "y" && symbol != "z") {
        return false;
    }

    if (!positionCached) {
        transform.transform(transform.integrationPoint(), position);
        positionCached = true;
    }

    if (symbol == "x") {
        value = position.x();
        return true;
    }
    if (symbol == "y") {
        value = position.y();
        return true;
    }

    value = position.z();
    return true;
}

class RuntimeScalarExpressionCoefficient final : public Coefficient {
public:
    RuntimeScalarExpressionCoefficient(std::string expression,
                                       const CaseDefinition& caseDef,
                                       ExternalRuntimeSymbolResolver externalResolver)
        : program_(expression, buildSymbolConfig(expression, caseDef)),
          externalResolver_(std::move(externalResolver)) {
    }

    void eval(ElementTransform& transform, Real& result, Real t = 0.0) const override {
        bool positionCached = false;
        Vector3 position = Vector3::Zero();

        RuntimeSymbolResolver resolver =
            [&](std::string_view symbol, double& value) {
                if (resolveBuiltInSymbol(symbol, transform, t, positionCached, position, value)) {
                    return true;
                }
                if (externalResolver_) {
                    return externalResolver_(symbol, transform, t, value);
                }
                return false;
            };

        result = program_.evaluate(resolver);
    }

private:
    CompiledScalarRuntimeProgram program_;
    ExternalRuntimeSymbolResolver externalResolver_;
};

class RuntimeMatrixExpressionCoefficient final : public MatrixCoefficient {
public:
    RuntimeMatrixExpressionCoefficient(std::string expression,
                                       const CaseDefinition& caseDef,
                                       ExternalRuntimeSymbolResolver externalResolver)
        : program_(expression, buildSymbolConfig(expression, caseDef)),
          externalResolver_(std::move(externalResolver)) {
    }

    void eval(ElementTransform& transform, Matrix3& result, Real t = 0.0) const override {
        bool positionCached = false;
        Vector3 position = Vector3::Zero();

        RuntimeSymbolResolver resolver =
            [&](std::string_view symbol, double& value) {
                if (resolveBuiltInSymbol(symbol, transform, t, positionCached, position, value)) {
                    return true;
                }
                if (externalResolver_) {
                    return externalResolver_(symbol, transform, t, value);
                }
                return false;
            };

        result = program_.evaluate(resolver);
    }

private:
    CompiledMatrixRuntimeProgram program_;
    ExternalRuntimeSymbolResolver externalResolver_;
};

}  // namespace

std::unique_ptr<Coefficient> createRuntimeScalarExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExternalRuntimeSymbolResolver externalResolver) {
    return std::make_unique<RuntimeScalarExpressionCoefficient>(
        std::move(expression),
        caseDef,
        std::move(externalResolver));
}

std::unique_ptr<MatrixCoefficient> createRuntimeMatrixExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExternalRuntimeSymbolResolver externalResolver) {
    return std::make_unique<RuntimeMatrixExpressionCoefficient>(
        std::move(expression),
        caseDef,
        std::move(externalResolver));
}

}  // namespace mpfem
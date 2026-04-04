#include "problem/expression_coefficient_factory.hpp"

#include "expr/runtime_program.hpp"
#include "expr/symbol_scanner.hpp"
#include "fe/element_transform.hpp"

#include <string>
#include <utility>
#include <vector>

namespace mpfem {
    namespace {

        RuntimeSymbolConfig buildSymbolConfig(const std::string& expression, const CaseDefinition& caseDef)
        {
            RuntimeSymbolConfig config;
            const std::vector<std::string> symbols = collectExpressionSymbols(expression);

            config.constants.reserve(symbols.size());
            config.dynamicSymbols.reserve(symbols.size());

            for (const std::string& symbol : symbols) {
                if (caseDef.hasVariable(symbol)) {
                    config.constants.emplace_back(symbol, caseDef.getVariable(symbol));
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
            double& value)
        {
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

        std::uint64_t hashStringTag(const std::string& text)
        {
            constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
            constexpr std::uint64_t kFnvPrime = 1099511628211ull;
            std::uint64_t h = kFnvOffset;
            for (unsigned char c : text) {
                h ^= static_cast<std::uint64_t>(c);
                h *= kFnvPrime;
            }
            return h;
        }

        std::uint64_t computeRuntimeTag(const std::vector<std::string>& dynamicSymbols,
            std::uint64_t baseTag,
            const ExternalRuntimeStateTagResolver& externalStateResolver)
        {
            std::uint64_t tag = baseTag;
            for (const std::string& symbol : dynamicSymbols) {
                if (symbol == "x" || symbol == "y" || symbol == "z") {
                    continue;
                }
                if (symbol == "t") {
                    return DynamicCoefficientTag;
                }
                if (!externalStateResolver) {
                    return DynamicCoefficientTag;
                }

                const std::uint64_t depTag = externalStateResolver(symbol);
                tag = combineTag(tag, depTag);
                if (tag == DynamicCoefficientTag) {
                    return tag;
                }
            }
            return tag;
        }

        template <typename BaseCoefficientT, typename ProgramT, typename ResultT>
        class RuntimeExpressionCoefficient final : public BaseCoefficientT {
        public:
            RuntimeExpressionCoefficient(std::string expression,
                const CaseDefinition& caseDef,
                RuntimeExpressionResolvers resolvers)
                : expressionTag_(hashStringTag(expression)), resolvers_(std::move(resolvers))
            {
                RuntimeSymbolConfig config = buildSymbolConfig(expression, caseDef);
                dynamicSymbols_ = config.dynamicSymbols;
                program_ = ProgramT(expression, std::move(config));
            }

            void eval(ElementTransform& transform, ResultT& result, Real t = 0.0) const override
            {
                bool positionCached = false;
                Vector3 position = Vector3::Zero();

                RuntimeSymbolResolver resolver =
                    [&](std::string_view symbol, double& value) {
                        if (resolveBuiltInSymbol(symbol, transform, t, positionCached, position, value)) {
                            return true;
                        }
                        if (resolvers_.symbolResolver) {
                            return resolvers_.symbolResolver(symbol, transform, t, value);
                        }
                        return false;
                    };

                result = program_.evaluate(resolver);
            }

            std::uint64_t stateTag() const override
            {
                return computeRuntimeTag(dynamicSymbols_, expressionTag_, resolvers_.stateTagResolver);
            }

        private:
            ProgramT program_ {"0", RuntimeSymbolConfig {}};
            std::uint64_t expressionTag_ = DynamicCoefficientTag;
            std::vector<std::string> dynamicSymbols_;
            RuntimeExpressionResolvers resolvers_;
        };

        using RuntimeScalarExpressionCoefficient = RuntimeExpressionCoefficient<Coefficient, CompiledScalarRuntimeProgram, Real>;
        using RuntimeMatrixExpressionCoefficient = RuntimeExpressionCoefficient<MatrixCoefficient, CompiledMatrixRuntimeProgram, Matrix3>;

    } // namespace

    std::unique_ptr<Coefficient> createRuntimeScalarExpressionCoefficient(
        std::string expression,
        const CaseDefinition& caseDef,
        RuntimeExpressionResolvers resolvers)
    {
        return std::make_unique<RuntimeScalarExpressionCoefficient>(
            std::move(expression),
            caseDef,
            std::move(resolvers));
    }

    std::unique_ptr<MatrixCoefficient> createRuntimeMatrixExpressionCoefficient(
        std::string expression,
        const CaseDefinition& caseDef,
        RuntimeExpressionResolvers resolvers)
    {
        return std::make_unique<RuntimeMatrixExpressionCoefficient>(
            std::move(expression),
            caseDef,
            std::move(resolvers));
    }

} // namespace mpfem
#ifndef MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP
#define MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP

#include "core/types.hpp"
#include "fe/coefficient.hpp"
#include "model/case_definition.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>

namespace mpfem {

using ExternalRuntimeSymbolResolver =
    std::function<bool(std::string_view, ElementTransform&, Real, double&)>;

using ExternalRuntimeStateTagResolver =
    std::function<std::uint64_t(std::string_view)>;

struct RuntimeExpressionResolvers {
    ExternalRuntimeSymbolResolver symbolResolver;
    ExternalRuntimeStateTagResolver stateTagResolver;
};

std::unique_ptr<Coefficient> createRuntimeScalarExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    RuntimeExpressionResolvers resolvers = {});

std::unique_ptr<MatrixCoefficient> createRuntimeMatrixExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    RuntimeExpressionResolvers resolvers = {});

}  // namespace mpfem

#endif  // MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP
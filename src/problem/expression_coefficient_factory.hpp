#ifndef MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP
#define MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP

#include "core/types.hpp"
#include "fe/coefficient.hpp"
#include "model/case_definition.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace mpfem {

using ExternalRuntimeSymbolResolver =
    std::function<bool(std::string_view, ElementTransform&, Real, double&)>;

std::unique_ptr<Coefficient> createRuntimeScalarExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExternalRuntimeSymbolResolver externalResolver);

std::unique_ptr<MatrixCoefficient> createRuntimeMatrixExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExternalRuntimeSymbolResolver externalResolver);

}  // namespace mpfem

#endif  // MPFEM_PROBLEM_EXPRESSION_COEFFICIENT_FACTORY_HPP

#ifndef MPFEM_COMPILED_EXPRESSION_COEFFICIENT_HPP
#define MPFEM_COMPILED_EXPRESSION_COEFFICIENT_HPP

#include "core/types.hpp"
#include "fe/coefficient.hpp"
#include "model/case_definition.hpp"

#include <functional>
#include <memory>
#include <string>

namespace mpfem {

struct ExpressionFieldAccessors {
    std::function<bool(ElementTransform&, Real, Real&)> sampleTemperature;
    std::function<bool(ElementTransform&, Real, Real&)> samplePotential;
};

std::unique_ptr<Coefficient> createCompiledScalarExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExpressionFieldAccessors fieldAccessors);

std::unique_ptr<MatrixCoefficient> createCompiledMatrixExpressionCoefficient(
    std::string expression,
    const CaseDefinition& caseDef,
    ExpressionFieldAccessors fieldAccessors);

}  // namespace mpfem

#endif  // MPFEM_COMPILED_EXPRESSION_COEFFICIENT_HPP

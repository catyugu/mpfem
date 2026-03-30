#ifndef MPFEM_EXPRESSION_SYMBOL_USAGE_HPP
#define MPFEM_EXPRESSION_SYMBOL_USAGE_HPP

#include "model/case_definition.hpp"

#include <string>
#include <vector>

namespace mpfem {

struct ExpressionSymbolUsage {
    bool useTime = false;
    bool useSpace = false;
    bool useTemperature = false;
    bool usePotential = false;
    std::vector<std::string> caseVariables;
};

ExpressionSymbolUsage analyzeExpressionSymbolUsage(const std::string& expression,
                                                   const CaseDefinition& caseDef);

}  // namespace mpfem

#endif  // MPFEM_EXPRESSION_SYMBOL_USAGE_HPP

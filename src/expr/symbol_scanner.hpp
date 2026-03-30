#ifndef MPFEM_EXPR_SYMBOL_SCANNER_HPP
#define MPFEM_EXPR_SYMBOL_SCANNER_HPP

#include <string>
#include <vector>

namespace mpfem {

std::vector<std::string> collectExpressionSymbols(const std::string& expression);

}  // namespace mpfem

#endif  // MPFEM_EXPR_SYMBOL_SCANNER_HPP

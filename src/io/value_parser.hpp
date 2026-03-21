#ifndef MPFEM_VALUE_PARSER_HPP
#define MPFEM_VALUE_PARSER_HPP

#include "core/types.hpp"
#include <string>
#include <map>
#include <optional>

namespace mpfem {

/**
 * @brief Parses numeric values from COMSOL-like value strings.
 */
class ValueParser {
public:
    /// Parses first scalar number from text such as "20[mV]"
    static bool parseFirstNumber(const std::string& text, double& value);
    
    /// Parse a value string and resolve variables
    static bool parseWithVariables(const std::string& text,
                                   const std::map<std::string, double>& variables,
                                   double& value);
    
    /// Parse a value string that may be a variable reference
    static double eval(const std::string& text,
                       const std::map<std::string, double>& variables);
    
    /// Parse matrix from format like "{'400','0','0','0','400','0','0','0','400'}"
    /// Returns nullopt if not a matrix format, returns diagonal matrix if scalar
    static std::optional<Matrix3> parseMatrix(const std::string& text);

private:
    static std::string trim(const std::string& str);
};

}  // namespace mpfem

#endif  // MPFEM_VALUE_PARSER_HPP
#ifndef MPFEM_VALUE_PARSER_HPP
#define MPFEM_VALUE_PARSER_HPP

#include <string>
#include <map>

namespace mpfem {

/**
 * @brief Parses numeric values from COMSOL-like value strings.
 * 
 * Supports formats like:
 * - "20[mV]" -> 20.0
 * - "9[cm]" -> 9.0
 * - "0.02" -> 0.02
 */
class ValueParser {
public:
    /**
     * @brief Parses first scalar number from text such as "20[mV]".
     * @param text Input text.
     * @param value Parsed numeric value.
     * @return True when parsing succeeds, false otherwise.
     */
    static bool parseFirstNumber(const std::string& text, double& value);

    /**
     * @brief Parse a value string and resolve variables.
     * @param text Input text (may contain variable references).
     * @param variables Map of variable name to value.
     * @param value Output value.
     * @return True if parsing succeeds.
     */
    static bool parseWithVariables(const std::string& text,
                                   const std::map<std::string, double>& variables,
                                   double& value);

    /**
     * @brief Parse a value string that may be a variable reference.
     * @param text Input text.
     * @param variables Map of variable name to value.
     * @return Parsed value, or 0.0 if parsing fails.
     */
    static double eval(const std::string& text,
                       const std::map<std::string, double>& variables);

private:
    static std::string trim(const std::string& str);
};

}  // namespace mpfem

#endif  // MPFEM_VALUE_PARSER_HPP

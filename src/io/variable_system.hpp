/**
 * @file variable_system.hpp
 * @brief Variable system with unit conversion support
 */

#ifndef MPFEM_IO_VARIABLE_SYSTEM_HPP
#define MPFEM_IO_VARIABLE_SYSTEM_HPP

#include "core/types.hpp"
#include <string>
#include <unordered_map>
#include <optional>
#include <functional>

namespace mpfem {

/**
 * @brief Result of parsing a value string with optional unit
 */
struct ParsedValue {
    Scalar value;           ///< Numeric value in SI units
    std::string unit;       ///< Original unit string (empty if no unit)
    bool valid;             ///< Whether parsing succeeded
    std::string error;      ///< Error message if parsing failed
    
    ParsedValue() : value(0), valid(false) {}
    ParsedValue(Scalar v, const std::string& u = "") 
        : value(v), unit(u), valid(true) {}
};

/**
 * @brief Unit conversion factor to SI base units
 */
struct UnitInfo {
    Scalar factor;          ///< Multiplicative factor to convert to SI
    Scalar offset;          ///< Additive offset (for temperature conversions)
    std::string si_unit;    ///< SI base unit name
    
    UnitInfo(Scalar f = 1.0, Scalar o = 0.0, const std::string& si = "")
        : factor(f), offset(o), si_unit(si) {}
};

/**
 * @brief Variable system managing parameters with unit conversion
 * 
 * Supports parsing values like "9[cm]", "20[mV]", "5[W/m^2/K]"
 * and converting them to SI units.
 */
class VariableSystem {
public:
    VariableSystem();
    ~VariableSystem() = default;
    
    /**
     * @brief Parse and add a variable from string like "9[cm]"
     * @param name Variable name
     * @param value_str Value string with optional unit in brackets
     * @return true if successful
     */
    bool parse_variable(const std::string& name, const std::string& value_str);
    
    /**
     * @brief Add a variable with pre-computed SI value
     * @param name Variable name
     * @param value Value in SI units
     */
    void set_variable(const std::string& name, Scalar value);
    
    /**
     * @brief Get variable value in SI units
     * @param name Variable name
     * @return Value in SI units, or 0 if not found
     */
    Scalar get(const std::string& name) const;
    
    /**
     * @brief Check if variable exists
     */
    bool has(const std::string& name) const;
    
    /**
     * @brief Get all variable names
     */
    std::vector<std::string> variable_names() const;
    
    /**
     * @brief Parse a value string and convert to SI units
     * @param value_str Value string like "9[cm]", "293.15[K]", or "0.02"
     * @return Parsed result
     */
    ParsedValue parse_value(const std::string& value_str) const;
    
    /**
     * @brief Convert value from one unit to another
     * @param value Value in source unit
     * @param from_unit Source unit
     * @param to_unit Target unit
     * @return Converted value
     */
    Scalar convert(Scalar value, const std::string& from_unit, 
                   const std::string& to_unit) const;
    
    /**
     * @brief Convert value to SI units
     * @param value Value in given unit
     * @param unit Unit string
     * @return Value in SI units
     */
    Scalar to_si(Scalar value, const std::string& unit) const;
    
    /**
     * @brief Evaluate expression with variable substitution
     * @param expr Expression like "Vtot / L" or "2 * pi * rad_1"
     * @return Evaluated result
     */
    Scalar evaluate(const std::string& expr) const;
    
    /**
     * @brief Substitute variables in expression string
     * @param expr Expression with variable names
     * @return Expression with variables replaced by their SI values
     */
    std::string substitute_variables(const std::string& expr) const;
    
    /**
     * @brief Clear all variables
     */
    void clear();
    
    /**
     * @brief Get unit info for a unit string
     */
    std::optional<UnitInfo> get_unit_info(const std::string& unit) const;
    
    /**
     * @brief Register a custom unit
     */
    void register_unit(const std::string& unit, const UnitInfo& info);

private:
    std::unordered_map<std::string, Scalar> variables_;
    std::unordered_map<std::string, std::string> original_units_;
    std::unordered_map<std::string, UnitInfo> unit_registry_;
    
    void initialize_default_units();
    
    /**
     * @brief Parse unit string and extract unit info
     * @param unit_str Unit string like "cm", "mV", "W/m^2/K"
     * @return Unit conversion info
     */
    std::optional<UnitInfo> parse_unit(const std::string& unit_str) const;
    
    /**
     * @brief Split compound unit into base units
     * @param unit Compound unit like "W/m^2/K"
     * @return Vector of (unit, power) pairs
     */
    std::vector<std::pair<std::string, int>> split_compound_unit(
        const std::string& unit) const;
};

} // namespace mpfem

#endif // MPFEM_IO_VARIABLE_SYSTEM_HPP

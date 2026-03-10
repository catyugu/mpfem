/**
 * @file variable_system.cpp
 * @brief Implementation of variable system with unit conversion
 */

#include "variable_system.hpp"
#include "core/logger.hpp"
#include <sstream>
#include <regex>
#include <cmath>
#include <cctype>

constexpr double M_PI = 3.14159265358979323846;

namespace mpfem {

VariableSystem::VariableSystem() {
    initialize_default_units();
}

void VariableSystem::initialize_default_units() {
    // SI prefixes
    // Length units (base: m)
    unit_registry_["m"] = UnitInfo(1.0, 0.0, "m");
    unit_registry_["km"] = UnitInfo(1e3, 0.0, "m");
    unit_registry_["cm"] = UnitInfo(1e-2, 0.0, "m");
    unit_registry_["mm"] = UnitInfo(1e-3, 0.0, "m");
    unit_registry_["um"] = UnitInfo(1e-6, 0.0, "m");
    unit_registry_["nm"] = UnitInfo(1e-9, 0.0, "m");
    
    // Time units (base: s)
    unit_registry_["s"] = UnitInfo(1.0, 0.0, "s");
    unit_registry_["ms"] = UnitInfo(1e-3, 0.0, "s");
    unit_registry_["min"] = UnitInfo(60.0, 0.0, "s");
    unit_registry_["h"] = UnitInfo(3600.0, 0.0, "s");
    
    // Mass units (base: kg)
    unit_registry_["kg"] = UnitInfo(1.0, 0.0, "kg");
    unit_registry_["g"] = UnitInfo(1e-3, 0.0, "kg");
    unit_registry_["mg"] = UnitInfo(1e-6, 0.0, "kg");
    
    // Temperature units (base: K)
    unit_registry_["K"] = UnitInfo(1.0, 0.0, "K");
    unit_registry_["C"] = UnitInfo(1.0, 273.15, "K");
    unit_registry_["degC"] = UnitInfo(1.0, 273.15, "K");
    
    // Electric potential (base: V)
    unit_registry_["V"] = UnitInfo(1.0, 0.0, "V");
    unit_registry_["mV"] = UnitInfo(1e-3, 0.0, "V");
    unit_registry_["kV"] = UnitInfo(1e3, 0.0, "V");
    
    // Electric current (base: A)
    unit_registry_["A"] = UnitInfo(1.0, 0.0, "A");
    unit_registry_["mA"] = UnitInfo(1e-3, 0.0, "A");
    
    // Electric resistance (base: ohm)
    unit_registry_["ohm"] = UnitInfo(1.0, 0.0, "ohm");
    unit_registry_["kohm"] = UnitInfo(1e3, 0.0, "ohm");
    
    // Power (base: W)
    unit_registry_["W"] = UnitInfo(1.0, 0.0, "W");
    unit_registry_["mW"] = UnitInfo(1e-3, 0.0, "W");
    unit_registry_["kW"] = UnitInfo(1e3, 0.0, "W");
    
    // Energy (base: J)
    unit_registry_["J"] = UnitInfo(1.0, 0.0, "J");
    unit_registry_["kJ"] = UnitInfo(1e3, 0.0, "J");
    unit_registry_["MJ"] = UnitInfo(1e6, 0.0, "J");
    
    // Force (base: N)
    unit_registry_["N"] = UnitInfo(1.0, 0.0, "N");
    unit_registry_["kN"] = UnitInfo(1e3, 0.0, "N");
    unit_registry_["MN"] = UnitInfo(1e6, 0.0, "N");
    
    // Pressure/Stress (base: Pa)
    unit_registry_["Pa"] = UnitInfo(1.0, 0.0, "Pa");
    unit_registry_["kPa"] = UnitInfo(1e3, 0.0, "Pa");
    unit_registry_["MPa"] = UnitInfo(1e6, 0.0, "Pa");
    unit_registry_["GPa"] = UnitInfo(1e9, 0.0, "Pa");
    unit_registry_["bar"] = UnitInfo(1e5, 0.0, "Pa");
    
    // Frequency (base: Hz)
    unit_registry_["Hz"] = UnitInfo(1.0, 0.0, "Hz");
    unit_registry_["kHz"] = UnitInfo(1e3, 0.0, "Hz");
    unit_registry_["MHz"] = UnitInfo(1e6, 0.0, "Hz");
    
    // Dimensionless
    unit_registry_["1"] = UnitInfo(1.0, 0.0, "");
    unit_registry_["%"] = UnitInfo(0.01, 0.0, "");
    
    // Mathematical constants
    unit_registry_["pi"] = UnitInfo(M_PI, 0.0, "");
    
    // Compound units - heat transfer coefficient
    // W/(m^2*K) = W * m^-2 * K^-1
    // factor = 1.0 (already SI)
    
    // Electric conductivity: S/m = 1/(ohm*m)
    // Resistivity: ohm*m
    
    // Thermal conductivity: W/(m*K)
}

bool VariableSystem::parse_variable(const std::string& name, 
                                     const std::string& value_str) {
    ParsedValue result = parse_value(value_str);
    if (!result.valid) {
        MPFEM_ERROR("Failed to parse variable '" << name << "': " << result.error);
        return false;
    }
    
    variables_[name] = result.value;
    original_units_[name] = result.unit;
    
    MPFEM_INFO("Variable '" << name << "' = " << value_str << " -> " << result.value << " SI");
    return true;
}

void VariableSystem::set_variable(const std::string& name, Scalar value) {
    variables_[name] = value;
    original_units_[name] = "";
}

Scalar VariableSystem::get(const std::string& name) const {
    auto it = variables_.find(name);
    if (it == variables_.end()) {
        MPFEM_WARN("Variable '" << name << "' not found, returning 0");
        return 0.0;
    }
    return it->second;
}

bool VariableSystem::has(const std::string& name) const {
    return variables_.find(name) != variables_.end();
}

std::vector<std::string> VariableSystem::variable_names() const {
    std::vector<std::string> names;
    names.reserve(variables_.size());
    for (const auto& [name, _] : variables_) {
        names.push_back(name);
    }
    return names;
}

ParsedValue VariableSystem::parse_value(const std::string& value_str) const {
    ParsedValue result;
    
    // Pattern: number[unit] or just number
    // Examples: "9[cm]", "20[mV]", "293.15[K]", "0.02"
    std::regex pattern(R"(^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\[([^\]]+)\]\s*$)");
    std::regex pattern_no_unit(R"(^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$)");
    
    std::smatch match;
    
    if (std::regex_match(value_str, match, pattern)) {
        // Has unit
        Scalar numeric_value = std::stod(match[1].str());
        std::string unit = match[2].str();
        
        result.unit = unit;
        result.value = to_si(numeric_value, unit);
        result.valid = true;
    } else if (std::regex_match(value_str, match, pattern_no_unit)) {
        // No unit
        result.value = std::stod(match[1].str());
        result.valid = true;
    } else {
        result.error = "Invalid format: " + value_str;
        result.valid = false;
    }
    
    return result;
}

Scalar VariableSystem::convert(Scalar value, const std::string& from_unit,
                                const std::string& to_unit) const {
    // Convert to SI first, then to target unit
    auto from_info = get_unit_info(from_unit);
    auto to_info = get_unit_info(to_unit);
    
    if (!from_info || !to_info) {
        MPFEM_ERROR("Unknown unit in conversion: " << from_unit << " -> " << to_unit);
        return value;
    }
    
    // Convert from source to SI
    Scalar si_value = (value + from_info->offset) * from_info->factor;
    
    // Convert from SI to target
    Scalar result = si_value / to_info->factor - to_info->offset;
    
    return result;
}

Scalar VariableSystem::to_si(Scalar value, const std::string& unit) const {
    if (unit.empty()) {
        return value;
    }
    
    auto info = get_unit_info(unit);
    if (!info) {
        MPFEM_WARN("Unknown unit '" << unit << "', treating as dimensionless");
        return value;
    }
    
    // Handle compound units
    auto parts = split_compound_unit(unit);
    if (parts.size() > 1) {
        // Already calculated in get_unit_info for compound units
        return (value + info->offset) * info->factor;
    }
    
    return (value + info->offset) * info->factor;
}

std::optional<UnitInfo> VariableSystem::get_unit_info(const std::string& unit) const {
    // Direct lookup
    auto it = unit_registry_.find(unit);
    if (it != unit_registry_.end()) {
        return it->second;
    }
    
    // Try to parse compound unit
    return parse_unit(unit);
}

void VariableSystem::register_unit(const std::string& unit, const UnitInfo& info) {
    unit_registry_[unit] = info;
}

std::optional<UnitInfo> VariableSystem::parse_unit(const std::string& unit_str) const {
    // Handle compound units like "W/m^2/K", "ohm*m", "kg/m^3"
    // Format: unit1*unit2/unit3^power
    
    auto parts = split_compound_unit(unit_str);
    if (parts.empty()) {
        return std::nullopt;
    }
    
    Scalar total_factor = 1.0;
    Scalar total_offset = 0.0;
    bool first = true;
    
    for (const auto& [part, power] : parts) {
        auto it = unit_registry_.find(part);
        if (it == unit_registry_.end()) {
            MPFEM_INFO("Unknown unit part: " << part);
            return std::nullopt;
        }
        
        if (power > 0) {
            total_factor *= std::pow(it->second.factor, power);
        } else {
            total_factor *= std::pow(it->second.factor, power);  // negative power = division
        }
        
        // Only the first unit's offset applies (for temperature conversions)
        if (first && it->second.offset != 0.0) {
            total_offset = it->second.offset;
        }
        first = false;
    }
    
    return UnitInfo(total_factor, total_offset, "");
}

std::vector<std::pair<std::string, int>> VariableSystem::split_compound_unit(
    const std::string& unit) const {
    
    std::vector<std::pair<std::string, int>> parts;
    
    // Split by '/' first (denominator parts)
    std::string remaining = unit;
    bool in_denominator = false;
    
    // Parse format: unit1*unit2/unit3^2/unit4
    std::regex part_pattern(R"(([^*/^]+)(?:\^([+-]?\d+))?)");
    
    // First split by /
    size_t slash_pos = remaining.find('/');
    std::string numerator = (slash_pos == std::string::npos) ? remaining : remaining.substr(0, slash_pos);
    std::string denominator = (slash_pos == std::string::npos) ? "" : remaining.substr(slash_pos + 1);
    
    // Parse numerator
    std::istringstream num_stream(numerator);
    std::string part;
    while (std::getline(num_stream, part, '*')) {
        // Remove whitespace
        part.erase(0, part.find_first_not_of(" \t"));
        part.erase(part.find_last_not_of(" \t") + 1);
        
        if (part.empty()) continue;
        
        // Check for power
        int power = 1;
        size_t caret_pos = part.find('^');
        if (caret_pos != std::string::npos) {
            power = std::stoi(part.substr(caret_pos + 1));
            part = part.substr(0, caret_pos);
        }
        
        parts.push_back({part, power});
    }
    
    // Parse denominator
    if (!denominator.empty()) {
        std::istringstream den_stream(denominator);
        while (std::getline(den_stream, part, '*')) {
            part.erase(0, part.find_first_not_of(" \t"));
            part.erase(part.find_last_not_of(" \t") + 1);
            
            if (part.empty()) continue;
            
            int power = 1;
            size_t caret_pos = part.find('^');
            if (caret_pos != std::string::npos) {
                power = std::stoi(part.substr(caret_pos + 1));
                part = part.substr(0, caret_pos);
            }
            
            parts.push_back({part, -power});  // Negative power for denominator
        }
    }
    
    return parts;
}

Scalar VariableSystem::evaluate(const std::string& expr) const {
    // First substitute variables
    std::string substituted = substitute_variables(expr);
    
    // Simple expression evaluation
    // For now, just handle basic arithmetic
    // TODO: Use a proper expression parser for complex expressions
    
    try {
        // Try to parse as a simple number first
        size_t idx;
        Scalar result = std::stod(substituted, &idx);
        if (idx == substituted.length()) {
            return result;
        }
    } catch (...) {
        // Not a simple number, continue with expression parsing
    }
    
    // Basic expression evaluation using a simple recursive descent parser
    std::istringstream stream(substituted);
    Scalar result = 0.0;
    char op = '+';
    Scalar term;
    
    while (stream >> term >> op) {
        switch (op) {
            case '+': result += term; break;
            case '-': result -= term; break;
            case '*': result *= term; break;
            case '/': result /= term; break;
        }
    }
    
    // If we didn't parse anything, try reading the whole thing as a number
    if (stream.fail() && result == 0.0) {
        std::istringstream fresh_stream(substituted);
        fresh_stream >> result;
    }
    
    return result;
}

std::string VariableSystem::substitute_variables(const std::string& expr) const {
    std::string result = expr;
    
    // Find and replace all variable names
    // Sort by length (longest first) to avoid partial replacements
    std::vector<std::string> names = variable_names();
    std::sort(names.begin(), names.end(), 
              [](const std::string& a, const std::string& b) {
                  return a.length() > b.length();
              });
    
    for (const auto& name : names) {
        // Replace variable name with its value
        std::string value_str = std::to_string(variables_.at(name));
        
        // Use word boundary matching
        size_t pos = 0;
        while ((pos = result.find(name, pos)) != std::string::npos) {
            // Check if it's a whole word
            bool valid_start = (pos == 0 || !std::isalnum(result[pos-1]));
            bool valid_end = (pos + name.length() >= result.length() || 
                              !std::isalnum(result[pos + name.length()]));
            
            if (valid_start && valid_end) {
                result.replace(pos, name.length(), value_str);
                pos += value_str.length();
            } else {
                pos += name.length();
            }
        }
    }
    
    return result;
}

void VariableSystem::clear() {
    variables_.clear();
    original_units_.clear();
}

} // namespace mpfem

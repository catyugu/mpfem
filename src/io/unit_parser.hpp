#ifndef SRC_IO_UNIT_PARSER_HPP
#define SRC_IO_UNIT_PARSER_HPP

#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <cmath>

struct UnitParseResult {
    std::string_view expression;
    double multiplier;
};

class UnitRegistry {
public:
    static UnitRegistry& instance() {
        static UnitRegistry inst;
        return inst;
    }

    UnitParseResult stripUnit(std::string_view input) const {
        auto start = input.find('[');
        auto end = input.find(']');

        if (start == std::string_view::npos || end == std::string_view::npos || end <= start) {
            return {input, 1.0};
        }

        std::string_view expr = input.substr(0, start);
        std::string_view unit = input.substr(start + 1, end - start - 1);

        return {expr, getMultiplier(unit)};
    }

    double getMultiplier(std::string_view unit) const {
        std::string unitStr(unit);
        auto it = units.find(unitStr);
        if (it != units.end()) {
            return it->second;
        }
        return parseCompoundUnit(unit);
    }

private:
    std::unordered_map<std::string, double> units;

    UnitRegistry() {
        units.reserve(32);
        units["GPa"] = 1e9;
        units["MPa"] = 1e6;
        units["kPa"] = 1e3;
        units["Pa"] = 1.0;
        units["W/(m*K)"] = 1.0;
        units["W m^-1 K^-1"] = 1.0;
        units["J/(kg*K)"] = 1.0;
        units["J"] = 1.0;
        units["kJ"] = 1e3;
        units["W"] = 1.0;
        units["K"] = 1.0;
        units["kg"] = 1.0;
        units["g"] = 1e-3;
        units["m"] = 1.0;
        units["cm"] = 1e-2;
        units["mm"] = 1e-3;
        units["s"] = 1.0;
        units["S/m"] = 1.0;
        units["1/K"] = 1.0;
        units["1/m"] = 1.0;
        units["kg/m^3"] = 1.0;
        units["kg/m^2"] = 1.0;
        units["kg/m"] = 1.0;
    }

    double parseCompoundUnit(std::string_view unit) const {
        size_t divPos = std::string_view::npos;
        int depth = 0;

        for (size_t i = 0; i < unit.size(); ++i) {
            if (unit[i] == '(') depth++;
            else if (unit[i] == ')') depth--;
            else if (unit[i] == '/' && depth == 0) {
                divPos = i;
                break;
            }
        }

        if (divPos != std::string_view::npos) {
            std::string_view num = unit.substr(0, divPos);
            std::string_view den = unit.substr(divPos + 1);
            return parseCompoundUnit(num) / parseCompoundUnit(den);
        }

        return parseUnitTerms(unit);
    }

    double parseUnitTerms(std::string_view unit) const {
        size_t multPos = std::string_view::npos;
        int depth = 0;

        for (size_t i = 0; i < unit.size(); ++i) {
            if (unit[i] == '(') depth++;
            else if (unit[i] == ')') depth--;
            else if (unit[i] == '*' && depth == 0) {
                multPos = i;
                break;
            }
        }

        if (multPos != std::string_view::npos) {
            std::string_view left = unit.substr(0, multPos);
            std::string_view right = unit.substr(multPos + 1);
            return parseUnitTerms(left) * parseUnitTerms(right);
        }

        return parseUnitElement(unit);
    }

    double parseUnitElement(std::string_view elem) const {
        auto caretPos = elem.find('^');
        if (caretPos != std::string_view::npos) {
            std::string_view base = elem.substr(0, caretPos);
            std::string_view expStr = elem.substr(caretPos + 1);
            double exp = std::stod(std::string(expStr));
            return std::pow(getBaseMultiplier(base), exp);
        }
        return getBaseMultiplier(elem);
    }

    double getBaseMultiplier(std::string_view baseUnit) const {
        std::string unitStr(baseUnit);
        auto it = units.find(unitStr);
        if (it != units.end()) {
            return it->second;
        }
        throw std::invalid_argument("Unknown unit: " + unitStr);
    }

    UnitRegistry(const UnitRegistry&) = delete;
    UnitRegistry& operator=(const UnitRegistry&) = delete;
    UnitRegistry(UnitRegistry&&) = delete;
    UnitRegistry& operator=(UnitRegistry&&) = delete;
};

inline double parseSI(std::string_view input) {
    auto result = UnitRegistry::instance().stripUnit(input);
    return std::stod(std::string(result.expression)) * result.multiplier;
}

#endif

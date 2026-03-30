#ifndef SRC_IO_UNIT_PARSER_HPP
#define SRC_IO_UNIT_PARSER_HPP

#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <cctype>

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
        auto end = input.rfind(']');

        if (start == std::string_view::npos || end == std::string_view::npos || end <= start) {
            return {input, 1.0};
        }

        std::string_view expr = trimView(input.substr(0, start));
        std::string_view unit = trimView(input.substr(start + 1, end - start - 1));

        return {expr, getMultiplier(unit)};
    }

    double getMultiplier(std::string_view unit) const {
        unit = trimView(unit);
        if (unit.empty()) {
            return 1.0;
        }

        const std::string unitKey(unit);

        auto it = units.find(unitKey);
        if (it != units.end()) {
            return it->second;
        }

        thread_local std::unordered_map<std::string, double> compiledCache;
        auto cacheIt = compiledCache.find(unitKey);
        if (cacheIt != compiledCache.end()) {
            return cacheIt->second;
        }

        const double parsed = parseCompoundUnit(unit);
        compiledCache.emplace(unitKey, parsed);
        return parsed;
    }

private:
    std::unordered_map<std::string, double> units;

    static std::string_view trimView(std::string_view text) {
        size_t first = 0;
        while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first])) != 0) {
            ++first;
        }
        size_t last = text.size();
        while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])) != 0) {
            --last;
        }
        return text.substr(first, last - first);
    }

    static std::string_view stripOuterParens(std::string_view text) {
        text = trimView(text);
        while (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
            int depth = 0;
            bool wrapsAll = true;
            for (size_t i = 0; i < text.size(); ++i) {
                const char c = text[i];
                if (c == '(') {
                    ++depth;
                } else if (c == ')') {
                    --depth;
                    if (depth == 0 && i + 1 < text.size()) {
                        wrapsAll = false;
                        break;
                    }
                }
            }
            if (!wrapsAll || depth != 0) {
                break;
            }
            text = trimView(text.substr(1, text.size() - 2));
        }
        return text;
    }

    UnitRegistry() {
        units.reserve(40);
        units["GPa"] = 1e9;
        units["MPa"] = 1e6;
        units["kPa"] = 1e3;
        units["Pa"] = 1.0;
        units["V"] = 1.0;
        units["mV"] = 1e-3;
        units["W/(m*K)"] = 1.0;
        units["W m^-1 K^-1"] = 1.0;
        units["W/m^2/K"] = 1.0;
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
        unit = stripOuterParens(unit);
        if (unit.empty()) {
            return 1.0;
        }

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
            const std::string_view numerator = unit.substr(0, divPos);
            const std::string_view denominator = unit.substr(divPos + 1);
            return parseCompoundUnit(numerator) / parseCompoundUnit(denominator);
        }

        return parseUnitTerms(unit);
    }

    double parseUnitTerms(std::string_view unit) const {
        unit = stripOuterParens(unit);
        if (unit.empty()) {
            return 1.0;
        }

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
            const std::string_view left = unit.substr(0, multPos);
            const std::string_view right = unit.substr(multPos + 1);
            return parseUnitTerms(left) * parseUnitTerms(right);
        }

        return parseUnitElement(unit);
    }

    double parseUnitElement(std::string_view elem) const {
        elem = stripOuterParens(elem);
        auto caretPos = elem.find('^');
        if (caretPos != std::string_view::npos) {
            std::string_view base = trimView(elem.substr(0, caretPos));
            std::string_view expStr = trimView(elem.substr(caretPos + 1));
            if (expStr.empty()) {
                throw std::invalid_argument("Missing unit exponent in: " + std::string(elem));
            }
            double exp = std::stod(std::string(expStr));
            return std::pow(getBaseMultiplier(base), exp);
        }
        return getBaseMultiplier(elem);
    }

    double getBaseMultiplier(std::string_view baseUnit) const {
        baseUnit = trimView(baseUnit);
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

#include "expr/unit_parser.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mpfem {
namespace {

std::string_view trimView(std::string_view text) {
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

std::string_view stripOuterParens(std::string_view text) {
    text = trimView(text);
    while (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
        int depth = 0;
        bool wrapsAll = true;

        for (size_t i = 0; i < text.size(); ++i) {
            const char c = text[i];
            if (c == '(') {
                ++depth;
                continue;
            }
            if (c == ')') {
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

}  // namespace

struct UnitRegistry::Impl {
    std::unordered_map<std::string, double> units;

    Impl() {
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

    double getBaseMultiplier(std::string_view baseUnit) const {
        baseUnit = trimView(baseUnit);
        std::string unitKey(baseUnit);
        auto it = units.find(unitKey);
        if (it != units.end()) {
            return it->second;
        }

        throw std::invalid_argument("Unknown unit: " + unitKey);
    }

    double parseUnitElement(std::string_view text) const {
        text = stripOuterParens(text);

        auto caretPos = text.find('^');
        if (caretPos == std::string_view::npos) {
            return getBaseMultiplier(text);
        }

        std::string_view base = trimView(text.substr(0, caretPos));
        std::string_view exponentText = trimView(text.substr(caretPos + 1));
        if (exponentText.empty()) {
            throw std::invalid_argument("Missing unit exponent in: " + std::string(text));
        }

        const double exponent = std::stod(std::string(exponentText));
        return std::pow(getBaseMultiplier(base), exponent);
    }

    double parseUnitTerms(std::string_view text) const {
        text = stripOuterParens(text);
        if (text.empty()) {
            return 1.0;
        }

        size_t multiplyPos = std::string_view::npos;
        int depth = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '(') {
                ++depth;
                continue;
            }
            if (text[i] == ')') {
                --depth;
                continue;
            }
            if (text[i] == '*' && depth == 0) {
                multiplyPos = i;
                break;
            }
        }

        if (multiplyPos == std::string_view::npos) {
            return parseUnitElement(text);
        }

        const std::string_view left = text.substr(0, multiplyPos);
        const std::string_view right = text.substr(multiplyPos + 1);
        return parseUnitTerms(left) * parseUnitTerms(right);
    }

    double parseCompoundUnit(std::string_view text) const {
        text = stripOuterParens(text);
        if (text.empty()) {
            return 1.0;
        }

        size_t dividePos = std::string_view::npos;
        int depth = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '(') {
                ++depth;
                continue;
            }
            if (text[i] == ')') {
                --depth;
                continue;
            }
            if (text[i] == '/' && depth == 0) {
                dividePos = i;
                break;
            }
        }

        if (dividePos == std::string_view::npos) {
            return parseUnitTerms(text);
        }

        const std::string_view numerator = text.substr(0, dividePos);
        const std::string_view denominator = text.substr(dividePos + 1);
        return parseCompoundUnit(numerator) / parseCompoundUnit(denominator);
    }
};

UnitRegistry& UnitRegistry::instance() {
    static UnitRegistry registry;
    return registry;
}

UnitRegistry::UnitRegistry()
    : impl_(std::make_unique<Impl>()) {
}

UnitRegistry::~UnitRegistry() = default;

UnitParseResult UnitRegistry::stripUnit(std::string_view input) const {
    const size_t start = input.find('[');
    const size_t end = input.rfind(']');

    if (start == std::string_view::npos || end == std::string_view::npos || end <= start) {
        return {input, 1.0};
    }

    const std::string_view expression = trimView(input.substr(0, start));
    const std::string_view unit = trimView(input.substr(start + 1, end - start - 1));
    return {expression, getMultiplier(unit)};
}

double UnitRegistry::getMultiplier(std::string_view unit) const {
    unit = trimView(unit);
    if (unit.empty()) {
        return 1.0;
    }

    const std::string unitKey(unit);
    auto it = impl_->units.find(unitKey);
    if (it != impl_->units.end()) {
        return it->second;
    }

    thread_local std::unordered_map<std::string, double> cache;
    auto cacheIt = cache.find(unitKey);
    if (cacheIt != cache.end()) {
        return cacheIt->second;
    }

    const double parsed = impl_->parseCompoundUnit(unit);
    cache.emplace(unitKey, parsed);
    return parsed;
}

double parseSI(std::string_view input) {
    const UnitParseResult parsed = UnitRegistry::instance().stripUnit(input);
    return std::stod(std::string(parsed.expression)) * parsed.multiplier;
}

}  // namespace mpfem

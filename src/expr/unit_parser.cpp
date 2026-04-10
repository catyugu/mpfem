#include "expr/unit_parser.hpp"
#include "core/exception.hpp"
#include "core/string_utils.hpp"
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_map>

namespace mpfem {

namespace {

const std::unordered_map<std::string_view, Real>& getBaseUnits() {
    static const std::unordered_map<std::string_view, Real> units = {
        {"GPa", 1e9}, {"MPa", 1e6}, {"kPa", 1e3}, {"Pa", 1.0},
        {"V", 1.0}, {"mV", 1e-3},
        {"W/(m*K)", 1.0}, {"W m^-1 K^-1", 1.0}, {"W/m^2/K", 1.0},
        {"J/(kg*K)", 1.0}, {"J", 1.0}, {"kJ", 1e3},
        {"W", 1.0}, {"K", 1.0},
        {"kg", 1.0}, {"g", 1e-3},
        {"m", 1.0}, {"cm", 1e-2}, {"mm", 1e-3},
        {"s", 1.0}, {"S/m", 1.0},
        {"1/K", 1.0}, {"1/m", 1.0},
        {"kg/m^3", 1.0}, {"kg/m^2", 1.0}, {"kg/m", 1.0}
    };
    return units;
}

class UnitEvaluator {
    std::string_view text;
    size_t pos = 0;

public:
    explicit UnitEvaluator(std::string_view t) : text(t) {}

    Real parse() {
        Real res = parseExpr();
        skipSpace();
        if (pos < text.size()) {
            MPFEM_THROW(ArgumentException, "Unexpected trailing characters in unit: " + std::string(text));
        }
        return res;
    }

private:
    void skipSpace() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
    }

    bool match(char c) {
        skipSpace();
        if (pos < text.size() && text[pos] == c) {
            pos++;
            return true;
        }
        return false;
    }

    Real parseExpr() {
        Real val = parsePower();
        for (;;) {
            if (match('*')) {
                val *= parsePower();
            } else if (match('/')) {
                val /= parsePower();
            } else {
                break;
            }
        }
        return val;
    }

    Real parsePower() {
        Real val = parsePrimary();
        skipSpace();
        if (match('^')) {
            Real exp = parseNumber();
            val = std::pow(val, exp);
        }
        return val;
    }

    Real parsePrimary() {
        skipSpace();
        if (match('(')) {
            Real val = parseExpr();
            if (!match(')')) {
                MPFEM_THROW(ArgumentException, "Missing ')' in unit: " + std::string(text));
            }
            return val;
        }

        if (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '.' || text[pos] == '+' || text[pos] == '-')) {
            return parseNumber();
        }

        if (pos < text.size() && std::isalpha(text[pos])) {
            size_t start = pos;
            while (pos < text.size() && std::isalpha(text[pos])) pos++;
            std::string_view id = text.substr(start, pos - start);
            
            const auto& units = getBaseUnits();
            auto it = units.find(id);
            if (it != units.end()) {
                return it->second;
            }
            MPFEM_THROW(ArgumentException, "Unknown base unit: " + std::string(id));
        }

        MPFEM_THROW(ArgumentException, "Invalid unit syntax near: " + std::string(text.substr(pos)));
    }

    Real parseNumber() {
        skipSpace();
        size_t end;
        std::string sub = std::string(text.substr(pos));
        Real val = std::stod(sub, &end);
        pos += end;
        return val;
    }
};

} // namespace

Real parseUnit(std::string_view unit) {
    std::string trimmed = strings::trim(std::string(unit));
    if (trimmed.empty()) return 1.0;

    const auto& units = getBaseUnits();
    // 首先提供 O(1) 快速路径解析常见完整单位
    auto it = units.find(trimmed);
    if (it != units.end()) {
        return it->second;
    }

    // 否则退化到复合表达式处理
    UnitEvaluator eval(trimmed);
    return eval.parse();
}

Real parseSI(std::string_view input) {
    size_t bracket = input.find('[');
    if (bracket == std::string_view::npos) {
        return std::stod(std::string(input));
    }
    double val = std::stod(std::string(input.substr(0, bracket)));
    size_t end = input.find(']', bracket);
    if (end != std::string_view::npos) {
        val *= parseUnit(input.substr(bracket + 1, end - bracket - 1));
    }
    return val;
}

} // namespace mpfem
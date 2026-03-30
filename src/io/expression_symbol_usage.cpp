#include "io/expression_symbol_usage.hpp"

#include <cctype>
#include <string_view>
#include <unordered_set>

namespace mpfem {
namespace {

bool isIdentifierStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool isIdentifierChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool isExponentIdentifier(std::string_view text, size_t index) {
    if (index == 0 || index + 1 >= text.size()) {
        return false;
    }

    const char c = text[index];
    if (c != 'e' && c != 'E') {
        return false;
    }

    const char prev = text[index - 1];
    if (std::isdigit(static_cast<unsigned char>(prev)) == 0 && prev != '.') {
        return false;
    }

    const char next = text[index + 1];
    return std::isdigit(static_cast<unsigned char>(next)) != 0 || next == '+' || next == '-';
}

std::unordered_set<std::string> collectIdentifiers(std::string_view expression) {
    std::unordered_set<std::string> identifiers;

    size_t index = 0;
    while (index < expression.size()) {
        const char c = expression[index];
        if (!isIdentifierStart(c) || isExponentIdentifier(expression, index)) {
            ++index;
            continue;
        }

        const size_t begin = index;
        ++index;
        while (index < expression.size() && isIdentifierChar(expression[index])) {
            ++index;
        }

        size_t probe = index;
        while (probe < expression.size() && std::isspace(static_cast<unsigned char>(expression[probe])) != 0) {
            ++probe;
        }
        if (probe < expression.size() && expression[probe] == '(') {
            continue;
        }

        identifiers.emplace(expression.substr(begin, index - begin));
    }

    return identifiers;
}

}  // namespace

ExpressionSymbolUsage analyzeExpressionSymbolUsage(const std::string& expression,
                                                   const CaseDefinition& caseDef) {
    const auto identifiers = collectIdentifiers(expression);

    ExpressionSymbolUsage usage;
    usage.useTime = identifiers.count("t") > 0;
    usage.useSpace =
        identifiers.count("x") > 0 || identifiers.count("y") > 0 || identifiers.count("z") > 0;
    usage.useTemperature = identifiers.count("T") > 0;
    usage.usePotential = identifiers.count("V") > 0;

    usage.caseVariables.reserve(caseDef.variableMap_.size());
    for (const auto& [name, _] : caseDef.variableMap_) {
        if (identifiers.count(name) > 0) {
            usage.caseVariables.push_back(name);
        }
    }

    return usage;
}

}  // namespace mpfem

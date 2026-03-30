#include "expr/symbol_scanner.hpp"

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

std::string removeBracketedUnits(std::string_view text) {
    std::string output;
    output.reserve(text.size());

    int bracketDepth = 0;
    for (char c : text) {
        if (c == '[') {
            ++bracketDepth;
            continue;
        }
        if (c == ']') {
            if (bracketDepth > 0) {
                --bracketDepth;
                continue;
            }
        }
        if (bracketDepth == 0) {
            output.push_back(c);
        }
    }

    return output;
}

}  // namespace

std::vector<std::string> collectExpressionSymbols(const std::string& expression) {
    const std::string normalized = removeBracketedUnits(expression);

    std::vector<std::string> symbols;
    std::unordered_set<std::string> seen;

    size_t index = 0;
    while (index < normalized.size()) {
        const char c = normalized[index];
        if (!isIdentifierStart(c) || isExponentIdentifier(normalized, index)) {
            ++index;
            continue;
        }

        const size_t begin = index;
        ++index;
        while (index < normalized.size() && isIdentifierChar(normalized[index])) {
            ++index;
        }

        size_t probe = index;
        while (probe < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[probe])) != 0) {
            ++probe;
        }

        if (probe < normalized.size() && normalized[probe] == '(') {
            continue;
        }

        const std::string symbol = normalized.substr(begin, index - begin);
        if (seen.insert(symbol).second) {
            symbols.push_back(symbol);
        }
    }

    return symbols;
}

}  // namespace mpfem

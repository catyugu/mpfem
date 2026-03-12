#include "io/value_parser.hpp"

#include <cctype>
#include <cstdlib>

namespace mpfem {

std::string ValueParser::trim(const std::string& str) {
    size_t first = 0;
    while (first < str.size() && std::isspace(static_cast<unsigned char>(str[first]))) {
        ++first;
    }
    size_t last = str.size();
    while (last > first && std::isspace(static_cast<unsigned char>(str[last - 1]))) {
        --last;
    }
    return str.substr(first, last - first);
}

bool ValueParser::parseFirstNumber(const std::string& text, double& value) {
    value = 0.0;

    std::string token;
    for (size_t i = 0; i < text.size(); ++i) {
        const char current = text[i];
        const bool isNumericChar = std::isdigit(static_cast<unsigned char>(current)) != 0
                                   || current == '+'
                                   || current == '-'
                                   || current == '.'
                                   || current == 'e'
                                   || current == 'E';
        if (isNumericChar) {
            token.push_back(current);
            continue;
        }

        if (!token.empty()) {
            break;
        }
    }

    if (token.empty()) {
        return false;
    }

    char* endPtr = nullptr;
    value = std::strtod(token.c_str(), &endPtr);
    if (endPtr == token.c_str()) {
        return false;
    }

    return true;
}

bool ValueParser::parseWithVariables(const std::string& text,
                                     const std::map<std::string, double>& variables,
                                     double& value) {
    std::string trimmed = trim(text);

    // First, try to parse as a direct number
    if (parseFirstNumber(trimmed, value)) {
        return true;
    }

    // Try to look up as a variable name
    auto it = variables.find(trimmed);
    if (it != variables.end()) {
        value = it->second;
        return true;
    }

    // Check if it's a variable name followed by unit brackets
    // e.g., "Vtot" or "0[V]"
    size_t bracketPos = trimmed.find('[');
    if (bracketPos != std::string::npos) {
        std::string beforeBracket = trim(trimmed.substr(0, bracketPos));
        
        // Try as variable name first
        auto varIt = variables.find(beforeBracket);
        if (varIt != variables.end()) {
            value = varIt->second;
            return true;
        }

        // Try to parse as number with unit
        if (parseFirstNumber(beforeBracket, value)) {
            return true;
        }
    }

    return false;
}

double ValueParser::eval(const std::string& text,
                         const std::map<std::string, double>& variables) {
    double value = 0.0;
    parseWithVariables(text, variables, value);
    return value;
}

}  // namespace mpfem

#include "io/value_parser.hpp"
#include "core/string_utils.hpp"
#include <cctype>
#include <cstdlib>
#include <vector>

namespace mpfem {

// Use shared trim from strings::
using strings::trim;

bool ValueParser::parseFirstNumber(const std::string& text, double& value) {
    value = 0.0;
    std::string token;
    for (size_t i = 0; i < text.size(); ++i) {
        const char current = text[i];
        const bool isNumericChar = std::isdigit(static_cast<unsigned char>(current)) != 0
                                   || current == '+' || current == '-'
                                   || current == '.' || current == 'e' || current == 'E';
        if (isNumericChar) {
            token.push_back(current);
            continue;
        }
        if (!token.empty()) break;
    }
    if (token.empty()) return false;
    char* endPtr = nullptr;
    value = std::strtod(token.c_str(), &endPtr);
    return endPtr != token.c_str();
}

bool ValueParser::parseWithVariables(const std::string& text,
                                     const std::map<std::string, double>& variables,
                                     double& value) {
    std::string trimmed = trim(text);
    if (parseFirstNumber(trimmed, value)) return true;
    
    auto it = variables.find(trimmed);
    if (it != variables.end()) { value = it->second; return true; }
    
    size_t bracketPos = trimmed.find('[');
    if (bracketPos != std::string::npos) {
        std::string beforeBracket = trim(trimmed.substr(0, bracketPos));
        auto varIt = variables.find(beforeBracket);
        if (varIt != variables.end()) { value = varIt->second; return true; }
        if (parseFirstNumber(beforeBracket, value)) return true;
    }
    return false;
}

double ValueParser::eval(const std::string& text,
                         const std::map<std::string, double>& variables) {
    double value = 0.0;
    parseWithVariables(text, variables, value);
    return value;
}

std::optional<Matrix3> ValueParser::parseMatrix(const std::string& text) {
    std::string trimmed = trim(text);
    
    // Check for matrix format: {'a','b',...}
    if (trimmed.size() < 2 || trimmed[0] != '{') {
        // Not a matrix format, try as scalar
        double scalar;
        if (parseFirstNumber(trimmed, scalar)) {
            return Matrix3::Identity() * scalar;  // Return diagonal matrix
        }
        return std::nullopt;
    }
    
    // Parse matrix elements
    std::vector<double> values;
    std::string token;
    bool inQuote = false;
    
    for (size_t i = 1; i < trimmed.size(); ++i) {
        char c = trimmed[i];
        if (c == '\'') {
            inQuote = !inQuote;
            if (!inQuote && !token.empty()) {
                double val;
                if (parseFirstNumber(token, val)) values.push_back(val);
                token.clear();
            }
        } else if (inQuote) {
            token.push_back(c);
        }
    }
    
    // Expected 9 values for 3x3 symmetric matrix
    if (values.size() == 9) {
        Matrix3 m;
        // COMSOL uses symmetric storage: (1,1), (2,1), (3,1), (1,2), (2,2), (3,2), (1,3), (2,3), (3,3)
        // or row-major: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        // From the XML pattern, it appears to be column-major
        m << values[0], values[3], values[6],
             values[1], values[4], values[7],
             values[2], values[5], values[8];
        return m;
    } else if (values.size() == 1) {
        return Matrix3::Identity() * values[0];
    }
    
    return std::nullopt;
}

}  // namespace mpfem
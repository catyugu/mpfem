#ifndef MPFEM_STRING_UTILS_HPP
#define MPFEM_STRING_UTILS_HPP

#include <string>
#include <cctype>

namespace mpfem {

/**
 * @brief String utility functions
 */
namespace strings {

/// Trim whitespace from both ends of a string
inline std::string trim(const std::string& str) {
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

}  // namespace strings

}  // namespace mpfem

#endif  // MPFEM_STRING_UTILS_HPP

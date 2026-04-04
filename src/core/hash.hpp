#ifndef MPFEM_HASH_HPP
#define MPFEM_HASH_HPP

#include "core/types.hpp"
#include <cstdint>
#include <cstring>
#include <limits>

namespace mpfem {

    // FNV-1a hash constants
    inline constexpr std::uint64_t kFNVOffsetBasis = 1469598103934665603ull;
    inline constexpr std::uint64_t kGoldenRatioPrime = 0x9e3779b97f4a7c15ull;

    // Sentinel for dynamic (non-cacheable) coefficients
    inline constexpr std::uint64_t DynamicCoefficientTag = std::numeric_limits<std::uint64_t>::max();

    // FNV-1a hash combine
    inline std::uint64_t combineTag(std::uint64_t seed, std::uint64_t value)
    {
        if (seed == DynamicCoefficientTag || value == DynamicCoefficientTag) {
            return DynamicCoefficientTag;
        }
        return seed ^ (value + kGoldenRatioPrime + (seed << 6) + (seed >> 2));
    }

    // Hash a real number for cache keying
    inline std::uint64_t hashRealTag(Real value)
    {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Real size mismatch in hashRealTag");
        std::memcpy(&bits, &value, sizeof(value));
        return kFNVOffsetBasis ^ bits;
    }

} // namespace mpfem

#endif // MPFEM_HASH_HPP
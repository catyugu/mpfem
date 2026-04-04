#ifndef MPFEM_ASSEMBLY_CHANGE_TRACKER_HPP
#define MPFEM_ASSEMBLY_CHANGE_TRACKER_HPP

#include "core/hash.hpp"
#include "fe/coefficient.hpp"

#include <concepts>
#include <cstdint>
#include <set>
#include <type_traits>
#include <utility>

namespace mpfem {

    template <typename T>
    concept HasStateTagMethod = requires(const T& value) {
        { value.stateTag() } -> std::convertible_to<std::uint64_t>;
    };

    inline std::uint64_t stateTagOf(std::uint64_t tag)
    {
        return tag;
    }

    template <typename IntegerT>
        requires(std::integral<std::remove_cvref_t<IntegerT>> && !std::same_as<std::remove_cvref_t<IntegerT>, bool> && !std::same_as<std::remove_cvref_t<IntegerT>, std::uint64_t>)
    inline std::uint64_t stateTagOf(IntegerT value)
    {
        return static_cast<std::uint64_t>(value);
    }

    inline std::uint64_t stateTagOf(const std::set<int>& ids)
    {
        std::uint64_t tag = kFNVOffsetBasis;
        for (int id : ids) {
            tag = combineTag(tag, stateTagOf(id));
        }
        return tag;
    }

    inline std::uint64_t stateTagOf(const Coefficient* coefficient)
    {
        return coefficient ? coefficient->stateTag() : DynamicCoefficientTag;
    }

    inline std::uint64_t stateTagOf(const VectorCoefficient* coefficient)
    {
        return coefficient ? coefficient->stateTag() : DynamicCoefficientTag;
    }

    inline std::uint64_t stateTagOf(const MatrixCoefficient* coefficient)
    {
        return coefficient ? coefficient->stateTag() : DynamicCoefficientTag;
    }

    template <typename FirstT, typename SecondT>
    inline std::uint64_t stateTagOf(const std::pair<FirstT, SecondT>& pairValue)
    {
        return combineTag(stateTagOf(pairValue.first), stateTagOf(pairValue.second));
    }

    template <HasStateTagMethod T>
    inline std::uint64_t stateTagOf(const T& value)
    {
        return static_cast<std::uint64_t>(value.stateTag());
    }

    template <typename Range>
    inline std::uint64_t stateTagOfRange(const Range& range)
    {
        std::uint64_t tag = kFNVOffsetBasis;
        for (const auto& entry : range) {
            tag = combineTag(tag, stateTagOf(entry));
        }
        return tag;
    }

    template <typename Range, typename Projector>
    inline std::uint64_t stateTagOfRange(const Range& range, Projector projector)
    {
        std::uint64_t tag = kFNVOffsetBasis;
        for (const auto& entry : range) {
            tag = combineTag(tag, stateTagOf(projector(entry)));
        }
        return tag;
    }

    struct AssemblyTagCache {
        bool initialized = false;
        std::uint64_t tag = 0;

        bool isUnchanged(std::uint64_t currentTag) const
        {
            if (!initialized) {
                return false;
            }
            if (currentTag == DynamicCoefficientTag || tag == DynamicCoefficientTag) {
                return false;
            }
            return currentTag == tag;
        }

        bool needsRebuild(std::uint64_t currentTag) const
        {
            return !isUnchanged(currentTag);
        }

        void update(std::uint64_t newTag)
        {
            tag = newTag;
            initialized = true;
        }
    };

} // namespace mpfem

#endif // MPFEM_ASSEMBLY_CHANGE_TRACKER_HPP
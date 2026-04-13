#ifndef MPFEM_TIME_SCHEME_HPP
#define MPFEM_TIME_SCHEME_HPP

namespace mpfem {

/**
 * @brief Time integration scheme types
 */
enum class TimeScheme {
    BDF1,  ///< first-order backward differentiation
    BDF2   ///< second-order backward differentiation
};

}  // namespace mpfem

#endif  // MPFEM_TIME_SCHEME_HPP

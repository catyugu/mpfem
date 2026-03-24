#ifndef MPFEM_TIME_SCHEME_HPP
#define MPFEM_TIME_SCHEME_HPP

namespace mpfem {

/**
 * @brief Time integration scheme types
 */
enum class TimeScheme {
    BackwardEuler,  ///< BDF1 - first-order backward Euler
    BDF2            ///< BDF2 - second-order backward differentiation
};

}  // namespace mpfem

#endif  // MPFEM_TIME_SCHEME_HPP

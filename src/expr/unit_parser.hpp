#ifndef MPFEM_EXPR_UNIT_PARSER_HPP
#define MPFEM_EXPR_UNIT_PARSER_HPP

#include "core/types.hpp"
#include <string_view>

namespace mpfem {

    /**
     * @brief 解析单位字符串（如 "kg*m/s^2", "W/(m*K)" 等），返回转换为国际标准单位 (SI) 的乘数。
     */
    Real parseUnit(std::string_view unit);

    /**
     * @brief 解析带有可选单位的纯数值（例如 "2.0[mm]" -> 0.002）
     */
    Real parseSI(std::string_view input);

} // namespace mpfem

#endif // MPFEM_EXPR_UNIT_PARSER_HPP
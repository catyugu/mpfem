#ifndef MPFEM_EXPR_UNIT_PARSER_HPP
#define MPFEM_EXPR_UNIT_PARSER_HPP

#include <memory>
#include <string_view>

namespace mpfem {

struct UnitParseResult {
    std::string_view expression;
    double multiplier;
};

class UnitRegistry {
public:
    static UnitRegistry& instance();

    UnitParseResult stripUnit(std::string_view input) const;
    double getMultiplier(std::string_view unit) const;

    UnitRegistry(const UnitRegistry&) = delete;
    UnitRegistry& operator=(const UnitRegistry&) = delete;
    UnitRegistry(UnitRegistry&&) = delete;
    UnitRegistry& operator=(UnitRegistry&&) = delete;

private:
    UnitRegistry();
    ~UnitRegistry();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

double parseSI(std::string_view input);

}  // namespace mpfem

#endif  // MPFEM_EXPR_UNIT_PARSER_HPP

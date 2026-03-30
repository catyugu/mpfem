#ifndef SRC_IO_UNIT_PARSER_HPP
#define SRC_IO_UNIT_PARSER_HPP

#include <string_view>
#include <memory>

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

#endif

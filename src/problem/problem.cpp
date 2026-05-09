#include "problem/problem.hpp"

namespace mpfem {

    Problem::~Problem() = default;

    void Problem::registerCaseDefinitionVariables()
    {
        for (const auto& entry : caseDef.getVariables()) {
            globalVariables_.define(entry.name, entry.valueText);
        }
    }

} // namespace mpfem

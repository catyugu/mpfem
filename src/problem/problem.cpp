#include "problem/problem.hpp"

#include "core/exception.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include <utility>

namespace mpfem {

    Problem::~Problem() = default;

    void Problem::registerCaseDefinitionVariables()
    {
        for (const auto& entry : caseDef.getVariables()) {
            globalVariables_.registerConstantExpression(entry.name, entry.valueText);
        }
    }

} // namespace mpfem

#ifndef MPFEM_PROBLEM_HPP
#define MPFEM_PROBLEM_HPP

#include "core/types.hpp"
#include "expr/variable_graph.hpp"
#include "field/grid_function.hpp"
#include "io/case_definition.hpp"
#include "io/material_database.hpp"
#include "mesh/mesh.hpp"
#include "physics/field_values.hpp"
#include <string>
#include <string_view>


namespace mpfem {

    class ElectrostaticsSolver;
    class HeatTransferSolver;
    class StructuralSolver;

    class Problem {
    public:
        /// @brief Unified variable manager for the entire problem
        VariableManager globalVariables_;

        void registerCaseDefinitionVariables();

        virtual ~Problem();
        virtual bool isTransient() const { return false; }

        // Physics presence queries
        bool hasElectrostatics() const { return electrostatics != nullptr; }
        bool hasHeatTransfer() const { return heatTransfer != nullptr; }
        bool hasStructural() const { return structural != nullptr; }
        bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
        bool hasThermalExpansion() const { return hasHeatTransfer() && hasStructural(); }
        bool isCoupled() const { return hasJouleHeating() || hasThermalExpansion(); }

        // Coupling parameters for coupled problems
        int couplingMaxIter = 15;
        Real couplingTol = 1e-4;

        std::unique_ptr<ElectrostaticsSolver> electrostatics;
        std::unique_ptr<HeatTransferSolver> heatTransfer;
        std::unique_ptr<StructuralSolver> structural;

        std::string caseName;
        std::unique_ptr<Mesh> mesh;
        MaterialDatabase materials;
        CaseDefinition caseDef;
        FieldValues fieldValues;
    };

} // namespace mpfem

#endif // MPFEM_PROBLEM_HPP

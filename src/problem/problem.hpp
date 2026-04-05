#ifndef MPFEM_PROBLEM_HPP
#define MPFEM_PROBLEM_HPP

#include "core/types.hpp"
#include "expr/variable_graph.hpp"
#include "fe/grid_function.hpp"
#include "mesh/mesh.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "physics/field_values.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpfem {

    class ElectrostaticsSolver;
    class HeatTransferSolver;
    class StructuralSolver;

    class Problem {
    protected:
        struct DomainPropertyKey {
            std::string property;
            int domainId = -1;

            bool operator==(const DomainPropertyKey& other) const
            {
                return domainId == other.domainId && property == other.property;
            }
        };

        struct DomainPropertyKeyHash {
            std::size_t operator()(const DomainPropertyKey& key) const
            {
                constexpr std::size_t kMix = 0x9e3779b97f4a7c15ull;
                const std::size_t h1 = std::hash<std::string> {}(key.property);
                const std::size_t h2 = std::hash<int> {}(key.domainId);
                return h1 ^ (h2 + kMix + (h1 << 6) + (h1 >> 2));
            }
        };

        std::unordered_map<DomainPropertyKey, const VariableNode*, DomainPropertyKeyHash> domainScalarNodes;
        std::unordered_map<DomainPropertyKey, const VariableNode*, DomainPropertyKeyHash> domainMatrixNodes;
        std::vector<std::unique_ptr<VariableNode>> ownedNodes;
        std::vector<std::unique_ptr<VariableManager>> ownedNodeManagers;

    public:
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

        const VariableNode* findDomainScalarNode(std::string_view property, int domainId) const;
        const VariableNode* findDomainMatrixNode(std::string_view property, int domainId) const;
        const VariableNode* setDomainScalarNode(std::string property,
                                                int domainId,
                                                const VariableNode* node);
        const VariableNode* setDomainMatrixNode(std::string property,
                                                int domainId,
                                                const VariableNode* node);

        const VariableNode* ownNode(std::unique_ptr<VariableNode> node);
        const VariableNode* ownScalarExpressionNode(std::string expression,
                                                    GraphRuntimeResolvers resolvers = {});
        const VariableNode* ownMatrixExpressionNode(std::string expression,
                                                    GraphRuntimeResolvers resolvers = {});
    };

} // namespace mpfem

#endif // MPFEM_PROBLEM_HPP

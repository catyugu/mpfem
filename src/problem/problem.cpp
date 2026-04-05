#include "problem/problem.hpp"

#include "core/exception.hpp"
#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include <utility>

namespace mpfem {

namespace {

void registerCaseConstants(VariableManager& manager, const CaseDefinition& caseDef)
{
    for (const auto& entry : caseDef.getVariables()) {
        manager.registerConstant(entry.name, entry.siValue);
    }
}

} // namespace

    Problem::~Problem() = default;

    const VariableNode* Problem::findDomainScalarNode(std::string_view property, int domainId) const
    {
        DomainPropertyKey key {std::string(property), domainId};
        auto it = domainScalarNodes.find(key);
        return it != domainScalarNodes.end() ? it->second : nullptr;
    }

    const VariableNode* Problem::findDomainMatrixNode(std::string_view property, int domainId) const
    {
        DomainPropertyKey key {std::string(property), domainId};
        auto it = domainMatrixNodes.find(key);
        return it != domainMatrixNodes.end() ? it->second : nullptr;
    }

    const VariableNode* Problem::setDomainScalarNode(std::string property,
                                                     int domainId,
                                                     const VariableNode* node)
    {
        DomainPropertyKey key {std::move(property), domainId};
        auto [it, _] = domainScalarNodes.insert_or_assign(std::move(key), node);
        return it->second;
    }

    const VariableNode* Problem::setDomainMatrixNode(std::string property,
                                                     int domainId,
                                                     const VariableNode* node)
    {
        DomainPropertyKey key {std::move(property), domainId};
        auto [it, _] = domainMatrixNodes.insert_or_assign(std::move(key), node);
        return it->second;
    }

    const VariableNode* Problem::ownNode(std::unique_ptr<VariableNode> node)
    {
        ownedNodes.push_back(std::move(node));
        return ownedNodes.back().get();
    }

    const VariableNode* Problem::ownScalarExpressionNode(std::string expression,
                                                         GraphRuntimeResolvers resolvers)
    {
        auto manager = std::make_unique<VariableManager>();
        registerCaseConstants(*manager, caseDef);
        constexpr const char* kNodeName = "$root_scalar";
        manager->registerScalarExpression(kNodeName, std::move(expression), std::move(resolvers));
        manager->compileGraph();
        const VariableNode* node = manager->get(kNodeName);
        if (!node) {
            MPFEM_THROW(ArgumentException, "Failed to build scalar expression node.");
        }
        ownedNodeManagers.push_back(std::move(manager));
        return node;
    }

    const VariableNode* Problem::ownMatrixExpressionNode(std::string expression,
                                                         GraphRuntimeResolvers resolvers)
    {
        auto manager = std::make_unique<VariableManager>();
        registerCaseConstants(*manager, caseDef);
        constexpr const char* kNodeName = "$root_matrix";
        manager->registerMatrixExpression(kNodeName, std::move(expression), std::move(resolvers));
        manager->compileGraph();
        const VariableNode* node = manager->get(kNodeName);
        if (!node) {
            MPFEM_THROW(ArgumentException, "Failed to build matrix expression node.");
        }
        ownedNodeManagers.push_back(std::move(manager));
        return node;
    }

} // namespace mpfem

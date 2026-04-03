#ifndef MPFEM_OPERATOR_CONFIG_HPP
#define MPFEM_OPERATOR_CONFIG_HPP

#include "operator/parameter_list.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mpfem {

    /**
     * @brief Configuration for a single operator with optional nested children.
     *
     * This is a data structure that represents the configuration of an operator
     * and its children. It is parsed from XML and consumed by OperatorFactory.
     */
    struct OperatorConfig {
        std::string type; // Operator type name: "CG", "GMRES", "Jacobi", etc.
        ParameterList params; // Operator-specific parameters

        // Named child configurations (named by role, not generic "sublist")
        // GMRES/CG: "Preconditioner"
        // AdditiveSchwarz: "LocalSolver", "CoarseSolver"
        // AMG: "Smoother"
        std::map<std::string, OperatorConfig> children;

        OperatorConfig() = default;
        OperatorConfig(std::string t) : type(std::move(t)) { }

        // Check if a child exists
        bool hasChild(const std::string& role) const
        {
            return children.find(role) != children.end();
        }

        // Get child config (throws if not found)
        const OperatorConfig& getChild(const std::string& role) const
        {
            auto it = children.find(role);
            if (it == children.end()) {
                throw std::runtime_error("OperatorConfig: missing child '" + role + "'");
            }
            return it->second;
        }

        // Get child config (returns nullptr if not found)
        const OperatorConfig* tryGetChild(const std::string& role) const
        {
            auto it = children.find(role);
            return (it != children.end()) ? &it->second : nullptr;
        }
    };

} // namespace mpfem

#endif
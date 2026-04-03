#ifndef MPFEM_OPERATOR_FACTORY_HPP
#define MPFEM_OPERATOR_FACTORY_HPP

#include "operator/cg_operator.hpp"
#include "operator/gauss_seidel_operator.hpp"
#include "operator/gmres_operator.hpp"
#include "operator/icc_operator.hpp"
#include "operator/ilu_operator.hpp"
#include "operator/jacobi_operator.hpp"
#include "operator/linear_operator.hpp"
#include "operator/operator_config.hpp"
#include "operator/sparse_lu_operator.hpp"
#include "operator/umfpack_operator.hpp"
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mpfem {

    /**
     * @brief Factory for creating LinearOperator instances from configuration.
     *
     * Unified factory that creates any operator based on type name or ParameterList.
     * Supports nested operators via OperatorConfig for recursive composition.
     *
     * Example XML configuration:
     * <Operator type="GMRES">
     *     <Parameters>
     *         <MaxIterations>500</MaxIterations>
     *         <Tolerance>1e-8</Tolerance>
     *     </Parameters>
     *     <Preconditioner>
     *         <Operator type="Jacobi"/>
     *     </Preconditioner>
     * </Operator>
     */
    class OperatorFactory {
    public:
        /**
         * @brief Create an operator from OperatorConfig recursively.
         *
         * This is the main entry point for XML parsing - handles nested operator creation.
         */
        static std::unique_ptr<LinearOperator> createRecursive(const OperatorConfig& config)
        {
            if (config.type.empty()) {
                throw std::runtime_error("OperatorFactory: empty operator type in config");
            }

            // Create the operator with its parameters
            auto op = createByType(config.type, config.params);

            // Handle child operators (preconditioner)
            std::string typeKey = normalizeToken(config.type);

            if (typeKey == "cg" || typeKey == "gmres") {
                if (auto* precondConfig = config.tryGetChild("Preconditioner")) {
                    auto precond = createRecursive(*precondConfig);
                    op->set_preconditioner(precond.release()); // Transfer ownership
                }
            }

            return op;
        }

        /**
         * @brief Create operator by type name with parameters.
         */
        static std::unique_ptr<LinearOperator> createByType(
            const std::string& type,
            const ParameterList& params = ParameterList())
        {
            std::string key = normalizeToken(type);

            if (key == "cg") {
                auto op = std::make_unique<CGOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "gmres") {
                auto op = std::make_unique<GMRESOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "umfpack") {
                auto op = std::make_unique<UMFPACKOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "sparselu" || key == "lu") {
                auto op = std::make_unique<SparseLUOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "jacobi" || key == "diagonal") {
                auto op = std::make_unique<JacobiOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "gaussseidel" || key == "gs") {
                auto op = std::make_unique<GaussSeidelOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "ilu") {
                auto op = std::make_unique<ILUOperator>();
                op->set_parameters(params);
                return op;
            }

            if (key == "icc") {
                auto op = std::make_unique<ICCOperator>();
                op->set_parameters(params);
                return op;
            }

            throw std::runtime_error("Unknown operator type: " + type);
        }

        /**
         * @brief Create operator from type name string.
         */
        static std::unique_ptr<LinearOperator> create(const std::string& type)
        {
            return createByType(type);
        }

        /**
         * @brief Create operator from type name string view.
         */
        static std::unique_ptr<LinearOperator> create(std::string_view type)
        {
            return createByType(std::string(type));
        }

        /**
         * @brief Get list of available operator names.
         */
        static std::vector<std::string> availableOperators()
        {
            return {
                "CG",
                "GMRES",
                "UMFPACK",
                "SparseLU",
                "Jacobi",
                "GaussSeidel",
                "ILU",
                "ICC"};
        }

    private:
        static std::string normalizeToken(std::string_view text)
        {
            std::string normalized;
            normalized.reserve(text.size());
            for (char ch : text) {
                const unsigned char value = static_cast<unsigned char>(ch);
                if (std::isalnum(value)) {
                    normalized.push_back(static_cast<char>(std::tolower(value)));
                }
            }
            return normalized;
        }
    };

} // namespace mpfem

#endif // MPFEM_OPERATOR_FACTORY_HPP

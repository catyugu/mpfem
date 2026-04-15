#include "solver/solver_config.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace mpfem {

    namespace detail {

        inline constexpr bool isMKLAvailable()
        {
#ifdef MPFEM_USE_MKL
            return true;
#else
            return false;
#endif
        }

        inline constexpr bool isSuiteSparseAvailable()
        {
#ifdef MPFEM_USE_UMFPACK
            return true;
#else
            return false;
#endif
        }

        static const OperatorMeta operatorRegistry[] = {
            {OperatorType::SparseLU, "SparseLU", "Eigen SparseLU direct solver", false, false, true},
            {OperatorType::Pardiso, "Pardiso", "MKL PARDISO direct solver", false, false, isMKLAvailable()},
            {OperatorType::Umfpack, "UMFPACK", "SuiteSparse UMFPACK direct solver", false, false, isSuiteSparseAvailable()},
            {OperatorType::CG, "CG", "Eigen Conjugate Gradient solver", true, true, true},
            {OperatorType::DGMRES, "DGMRES", "Eigen DGMRES solver", true, false, true},
            {OperatorType::Diagonal, "Diagonal", "Diagonal (Jacobi) preconditioner", false, false, true},
            {OperatorType::ICC, "ICC", "Incomplete Cholesky preconditioner", false, true, true},
            {OperatorType::ILU, "ILU", "Incomplete LU preconditioner", false, false, true},
            {OperatorType::AdditiveSchwarz, "AdditiveSchwarz", "Additive Schwarz domain decomposition", false, false, true},
        };

        static constexpr size_t operatorRegistrySize = sizeof(operatorRegistry) / sizeof(OperatorMeta);

    } // namespace detail

    const OperatorMeta& getOperatorMeta(OperatorType type)
    {
        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (detail::operatorRegistry[i].type == type) {
                return detail::operatorRegistry[i];
            }
        }
        throw std::runtime_error("Unknown operator type");
    }

    std::string_view operatorTypeName(OperatorType type)
    {
        return getOperatorMeta(type).name;
    }

    bool isOperatorAvailable(OperatorType type)
    {
        return getOperatorMeta(type).isAvailable;
    }

    std::vector<std::string> availableOperatorNames()
    {
        std::vector<std::string> result;
        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (detail::operatorRegistry[i].isAvailable) {
                result.emplace_back(detail::operatorRegistry[i].name);
            }
        }
        return result;
    }

    OperatorType operatorTypeFromName(std::string_view name)
    {
        auto iequals = [](std::string_view a, std::string_view b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
                    return false;
            }
            return true;
        };

        for (size_t i = 0; i < detail::operatorRegistrySize; ++i) {
            if (iequals(detail::operatorRegistry[i].name, name)) {
                return detail::operatorRegistry[i].type;
            }
        }
        throw std::runtime_error("Unknown operator type: " + std::string(name));
    }

    LinearOperatorConfig::LinearOperatorConfig(const LinearOperatorConfig& other)
        : type(other.type), parameters(other.parameters)
    {
        if (other.preconditioner) preconditioner = std::make_unique<LinearOperatorConfig>(*other.preconditioner);
        if (other.localSolver) localSolver = std::make_unique<LinearOperatorConfig>(*other.localSolver);
        if (other.coarseSolver) coarseSolver = std::make_unique<LinearOperatorConfig>(*other.coarseSolver);
        if (other.smoother) smoother = std::make_unique<LinearOperatorConfig>(*other.smoother);
    }

    LinearOperatorConfig& LinearOperatorConfig::operator=(const LinearOperatorConfig& other)
    {
        if (this == &other) return *this;
        type = other.type;
        parameters = other.parameters;
        preconditioner = other.preconditioner ? std::make_unique<LinearOperatorConfig>(*other.preconditioner) : nullptr;
        localSolver = other.localSolver ? std::make_unique<LinearOperatorConfig>(*other.localSolver) : nullptr;
        coarseSolver = other.coarseSolver ? std::make_unique<LinearOperatorConfig>(*other.coarseSolver) : nullptr;
        smoother = other.smoother ? std::make_unique<LinearOperatorConfig>(*other.smoother) : nullptr;
        return *this;
    }

} // namespace mpfem

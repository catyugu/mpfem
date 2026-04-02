#ifndef MPFEM_SOLVER_FACTORY_HPP
#define MPFEM_SOLVER_FACTORY_HPP

#include "solver_config.hpp"
#include "linear_solver.hpp"
#include "eigen_solver.hpp"
#include "pardiso_solver.hpp"
#include "umfpack_solver.hpp"
#include "core/logger.hpp"
#include <cctype>
#include <memory>
#include <string_view>

namespace mpfem {

// =============================================================================
// Solver Factory
// =============================================================================

/**
 * @brief Factory for creating linear solvers.
 * 
 * Design principles:
 * - No fallback logic - if solver is not available, throw exception
 * - Single entry point: create(const SolverConfig&)
 * - Configuration must specify an explicit solver type
 */
class SolverFactory {
public:
    static LinearSolverType linearSolverTypeFromName(std::string_view name) {
        const std::string key = normalizeToken(name);
        if (key == "cg") {
            return LinearSolverType::Eigen_CG;
        }
        if (key == "fgmres") {
            return LinearSolverType::Eigen_DGMRES;
        }
        if (key == "sparselu") {
            return LinearSolverType::Eigen_SparseLU;
        }
        if (key == "pardiso") {
            return LinearSolverType::MKL_Pardiso;
        }
        if (key == "umfpack") {
            return LinearSolverType::UMFPACK_LU;
        }

        throw std::runtime_error("Unsupported LinearSolver type: " + std::string(name));
    }

    static PreconditionerType preconditionerTypeFromName(std::string_view name) {
        const std::string key = normalizeToken(name);
        if (key == "none") {
            return PreconditionerType::None;
        }
        if (key == "diagonal") {
            return PreconditionerType::Diagonal;
        }
        if (key == "icc") {
            return PreconditionerType::ICC;
        }
        if (key == "ilu") {
            return PreconditionerType::ILU;
        }
        if (key == "additiveschwarz") {
            return PreconditionerType::AdditiveSchwarz;
        }
        if (key == "amg") {
            return PreconditionerType::AMG;
        }
        if (key == "gaussseidel") {
            return PreconditionerType::GaussSeidel;
        }

        throw std::runtime_error("Unsupported Preconditioner type: " + std::string(name));
    }

    /**
     * @brief Create a solver from configuration.
     * @param config Solver configuration with explicit type
     * @throws std::runtime_error if solver is not available
     */
    static std::unique_ptr<LinearSolver> create(const SolverConfig& config) {
        const LinearSolverType type = config.linearType;
        const PreconditionerType preconditioner = config.effectivePreconditioner().type;
        
        // Check availability
        const auto& meta = getLinearSolverMeta(type);
        if (!meta.isAvailable) {
            throw std::runtime_error(
                "Solver '" + std::string(meta.name) + "' is not available. "
                "Available solvers: " + joinSolverNames());
        }
        
        LOG_DEBUG << "Creating solver: " << meta.name
                  << ", preconditioner: " << preconditionerTypeName(preconditioner);
        
        // Create solver instance
        auto solver = createByType(type, preconditioner);
        
        // Apply configuration
        solver->setMaxIterations(config.maxIterations);
        solver->setTolerance(config.relativeTolerance);
        solver->setPrintLevel(config.printLevel);
        
        // Apply solver-specific configuration
        solver->applyConfig(config);
        
        return solver;
    }

private:
    static std::string normalizeToken(std::string_view text) {
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

    static std::unique_ptr<LinearSolver> createByType(LinearSolverType type, PreconditionerType preconditioner) {
        switch (type) {
            // Eigen solvers (always available)
            case LinearSolverType::Eigen_SparseLU:
                if (preconditioner != PreconditionerType::None) {
                    throw std::runtime_error("SparseLU does not support a preconditioner");
                }
                return std::make_unique<EigenSparseLUSolver>();
            case LinearSolverType::Eigen_CG:
                if (preconditioner == PreconditionerType::Diagonal || preconditioner == PreconditionerType::None) {
                    return std::make_unique<EigenCGJacobiSolver>();
                }
                if (preconditioner == PreconditionerType::ICC) {
                    return std::make_unique<EigenCGICCSolver>();
                }
                if (preconditioner == PreconditionerType::ILU) {
                    return std::make_unique<EigenCGILUSolver>();
                }
                break;
            case LinearSolverType::Eigen_DGMRES:
                if (preconditioner == PreconditionerType::ILU || preconditioner == PreconditionerType::None) {
                    return std::make_unique<EigenDGMRESILUSolver>();
                }
                break;

            // MKL PARDISO
            case LinearSolverType::MKL_Pardiso:
                if (preconditioner != PreconditionerType::None) {
                    throw std::runtime_error("Pardiso does not support a preconditioner");
                }
                return std::make_unique<PardisoSolver>();

            // SuiteSparse UMFPACK
            case LinearSolverType::UMFPACK_LU:
                if (preconditioner != PreconditionerType::None) {
                    throw std::runtime_error("UMFPACK does not support a preconditioner");
                }
                return std::make_unique<UmfpackSolver>();

            default:
                throw std::runtime_error("Unsupported solver type");
        }

        throw std::runtime_error(
            "Unsupported solver/preconditioner combination: solver='" +
            std::string(linearSolverTypeName(type)) +
            "', preconditioner='" +
            std::string(preconditionerTypeName(preconditioner)) + "'");
    }
    
    static std::string joinSolverNames() {
        auto names = availableLinearSolverNames();
        std::string result;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i > 0) result += ", ";
            result += names[i];
        }
        return result;
    }
};

}  // namespace mpfem

#endif  // MPFEM_SOLVER_FACTORY_HPP

#include "solver_factory.hpp"

namespace mpfem {

    std::unique_ptr<LinearOperator> OperatorFactory::create(const LinearOperatorConfig& config)
    {
        const auto& meta = getOperatorMeta(config.type);
        if (!meta.isAvailable) {
            throw std::runtime_error(
                "Operator '" + std::string(meta.name) + "' is not available.");
        }

        std::unique_ptr<LinearOperator> op = createByType(config.type);

        // Configure operator parameters via virtual configure() method (no dynamic_cast needed)
        op->configure(config);

        // Create and attach nested preconditioner
        if (config.preconditioner) {
            auto pc = create(*config.preconditioner);
            op->set_preconditioner(std::move(pc));
        }

        return op;
    }

    std::unique_ptr<LinearOperator> OperatorFactory::createByType(OperatorType type)
    {
        switch (type) {
        case OperatorType::SparseLU:
            return std::make_unique<EigenSparseLUOperator>();
        case OperatorType::CG:
            return std::make_unique<CgOperator>();
        case OperatorType::DGMRES:
            return std::make_unique<GmresOperator>();
        case OperatorType::Diagonal:
            return std::make_unique<DiagonalOperator>();
        case OperatorType::ICC:
            return std::make_unique<IccOperator>();
        case OperatorType::ILU:
            return std::make_unique<IluOperator>();
        case OperatorType::AdditiveSchwarz:
            return std::make_unique<AdditiveSchwarzOperator>();
        case OperatorType::Pardiso:
#ifdef MPFEM_USE_MKL
            return std::make_unique<PardisoSolver>();
#else
            throw std::runtime_error("PardisoOperator: MKL not available");
#endif
        case OperatorType::Umfpack:
#ifdef MPFEM_USE_UMFPACK
            return std::make_unique<UmfpackSolver>();
#else
            throw std::runtime_error("UmfpackOperator: SuiteSparse not available");
#endif
        default:
            throw std::runtime_error("Unsupported operator type");
        }
    }

} // namespace mpfem
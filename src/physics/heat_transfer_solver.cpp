#include "heat_transfer_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

namespace {

class ProductScalarNode final : public VariableNode {
public:
    ProductScalarNode(const VariableNode* lhs, const VariableNode* rhs)
        : lhs_(lhs), rhs_(rhs)
    {
        if (!lhs_ || !rhs_) {
            MPFEM_THROW(ArgumentException, "ProductScalarNode requires non-null inputs.");
        }
        if (lhs_->shape() != VariableShape::Scalar || rhs_->shape() != VariableShape::Scalar) {
            MPFEM_THROW(ArgumentException, "ProductScalarNode inputs must be scalar.");
        }
    }

    VariableShape shape() const override { return VariableShape::Scalar; }

    std::pair<int, int> dimensions() const override { return {1, 1}; }

    void evaluateBatch(const EvaluationContext& ctx, std::span<double> dest) const override
    {
        std::vector<double> lhsValues(dest.size());
        std::vector<double> rhsValues(dest.size());
        lhs_->evaluateBatch(ctx, std::span<double>(lhsValues.data(), lhsValues.size()));
        rhs_->evaluateBatch(ctx, std::span<double>(rhsValues.data(), rhsValues.size()));
        for (size_t i = 0; i < dest.size(); ++i) {
            dest[i] = lhsValues[i] * rhsValues[i];
        }
    }

    std::vector<const VariableNode*> dependencies() const override
    {
        return {lhs_, rhs_};
    }

private:
    const VariableNode* lhs_ = nullptr;
    const VariableNode* rhs_ = nullptr;
};

} // namespace

    bool HeatTransferSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialTemperature)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));

        fieldValues.createScalarField(FieldId::Temperature, fes_.get(), initialTemperature);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
        solver_ = SolverFactory::create(*solverConfig_);

        LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";

        // Assemble mass matrix if density and specific heat are set
        if (!massBindings_.empty()) {
            assembleMassMatrix();
        }

        return true;
    }

    void HeatTransferSolver::setThermalConductivity(const std::set<int>& domains, const VariableNode* k)
    {
        conductivityBindings_.push_back({domains, k});
    }

    void HeatTransferSolver::setHeatSource(const std::set<int>& domains, const VariableNode* Q)
    {
        heatSourceBindings_.push_back({domains, Q});
    }

    void HeatTransferSolver::setMassProperties(const std::set<int>& domains,
        const VariableNode* rho,
        const VariableNode* Cp)
    {
        MassBinding binding;
        binding.domains = domains;
        binding.density = rho;
        binding.specificHeat = Cp;
        massBindings_.push_back(std::move(binding));
        massMatrixAssembled_ = false;
    }

    void HeatTransferSolver::assembleMassMatrix()
    {
        if (massBindings_.empty()) {
            LOG_ERROR << "HeatTransferSolver: mass properties not set for mass matrix";
            return;
        }

        auto massAsm = std::make_unique<BilinearFormAssembler>(fes_.get());
        std::vector<std::unique_ptr<ProductScalarNode>> rhoCpNodes;
        rhoCpNodes.reserve(massBindings_.size());

        for (const auto& binding : massBindings_) {
            rhoCpNodes.push_back(std::make_unique<ProductScalarNode>(binding.density, binding.specificHeat));

            massAsm->addDomainIntegrator(
                std::make_unique<MassIntegrator>(rhoCpNodes.back().get()),
                binding.domains);
        }
        massAsm->assemble();

        massMatrix_ = std::move(massAsm->matrix());
        massMatrixAssembled_ = true;

        LOG_INFO << "HeatTransferSolver: mass matrix assembled (" << massMatrix_.rows() << "x" << massMatrix_.cols() << ")";
    }

    void HeatTransferSolver::addTemperatureBC(const std::set<int>& boundaryIds, const VariableNode* temperature)
    {
        temperatureBindings_.push_back({boundaryIds, temperature});
    }

    void HeatTransferSolver::addConvectionBC(const std::set<int>& boundaryIds, const VariableNode* h, const VariableNode* Tinf)
    {
        convectionBindings_.push_back({boundaryIds, h, Tinf});
    }

    void HeatTransferSolver::assemble()
    {
        ScopedTimer timer("HeatTransfer assemble");

        if (conductivityBindings_.empty()) {
            LOG_ERROR << "HeatTransferSolver: conductivity not set";
            return;
        }

        if (!massBindings_.empty()) {
            assembleMassMatrix();
        }
        matAsm_->clear();
        matAsm_->clearIntegrators();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(
                std::make_unique<DiffusionIntegrator>(binding.conductivity),
                binding.domains);
        }

        for (const auto& binding : convectionBindings_) {
            for (int bid : binding.boundaryIds) {
                matAsm_->addBoundaryIntegrator(std::make_unique<ConvectionMassIntegrator>(binding.h), bid);
            }
        }

        matAsm_->assemble();
        stiffnessMatrixBeforeBC_ = matAsm_->matrix();

        vecAsm_->clear();
        vecAsm_->clearIntegrators();

        for (const auto& binding : convectionBindings_) {
            for (int bid : binding.boundaryIds) {
                vecAsm_->addBoundaryIntegrator(std::make_unique<ConvectionLFIntegrator>(binding.h, binding.Tinf), bid);
            }
        }

        for (const auto& binding : heatSourceBindings_) {
            vecAsm_->addDomainIntegrator(
                std::make_unique<DomainLFIntegrator>(binding.source),
                binding.domains);
        }
        vecAsm_->assemble();
        rhsBeforeBC_ = vecAsm_->vector();

        // Flatten temperatureBindings_ to map for applyDirichletBC
        std::map<int, const VariableNode*> temperatureBCs;
        for (const auto& binding : temperatureBindings_) {
            for (int bid : binding.boundaryIds) {
                temperatureBCs[bid] = binding.temperature;
            }
        }
        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), this->field().values(), *fes_, *mesh_, temperatureBCs);
        matAsm_->finalize();
    }

    bool HeatTransferSolver::solveLinearSystem(SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!solver_) {
            LOG_ERROR << "HeatTransferSolver: solver not available";
            return false;
        }

        // Use A directly instead of copying to persistent buffer
        systemRhs_ = b;

        // Flatten temperatureBindings_ to map for applyDirichletBC
        std::map<int, const VariableNode*> temperatureBCs;
        for (const auto& binding : temperatureBindings_) {
            for (int bid : binding.boundaryIds) {
                temperatureBCs[bid] = binding.temperature;
            }
        }

        // Apply boundary conditions to the combined system (A is modified in-place)
        applyDirichletBC(A, systemRhs_, x, *fes_, *mesh_, temperatureBCs);
        A.makeCompressed();

        // Two-stage solve: setup then apply
        solver_->setup(&A);
        solver_->apply(systemRhs_, x);

        this->field().markUpdated();
        LOG_INFO << fieldName() << " linear solve converged in " << this->iterations() << " iterations";

        return true;
    }

} // namespace mpfem

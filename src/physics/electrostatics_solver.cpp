#include "electrostatics_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"

namespace mpfem {

    bool ElectrostaticsSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialPotential)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));

        fieldValues.createScalarField(FieldId::ElectricPotential, fes_.get(), initialPotential);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());

        LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
        return true;
    }

    void ElectrostaticsSolver::setElectricalConductivity(const std::set<int>& domains, const MatrixCoefficient* sigma)
    {
        conductivityBindings_.push_back({domains, sigma});
    }

    void ElectrostaticsSolver::addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage)
    {
        for (int bid : boundaryIds)
            voltageBCs_[bid] = voltage;
    }

    void ElectrostaticsSolver::assemble()
    {
        ScopedTimer timer("Electrostatics assemble");

        if (conductivityBindings_.empty()) {
            LOG_ERROR << "ElectrostaticsSolver: conductivity not set";
            return;
        }

        const std::uint64_t currentTag = combineTag(
            stateTagOfRange(conductivityBindings_),
            stateTagOfRange(voltageBCs_));
        if (assembledSystemState_.isUnchanged(currentTag)) {
            LOG_DEBUG << "Electrostatics assemble skipped (coefficients unchanged)";
            return;
        }

        clearAssemblers();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(binding.sigma), binding.domains);
        }
        matAsm_->assemble();
        vecAsm_->assemble();

        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, voltageBCs_);
        matAsm_->finalize();

        assembledSystemState_.update(currentTag);
    }

} // namespace mpfem

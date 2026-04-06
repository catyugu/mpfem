#include "electrostatics_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    bool ElectrostaticsSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialPotential)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));

        fieldValues.createScalarField("V", fes_.get(), initialPotential);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
        solver_ = SolverFactory::create(*solverConfig_);

        LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
        return true;
    }

    void ElectrostaticsSolver::setElectricalConductivity(const std::set<int>& domains, const VariableNode* sigma)
    {
        conductivityBindings_.push_back({domains, sigma});
    }

    void ElectrostaticsSolver::addVoltageBC(const std::set<int>& boundaryIds, const VariableNode* voltage)
    {
        voltageBindings_.push_back({boundaryIds, voltage});
    }

    void ElectrostaticsSolver::assemble()
    {
        ScopedTimer timer("Electrostatics assemble");

        if (conductivityBindings_.empty()) {
            LOG_ERROR << "ElectrostaticsSolver: conductivity not set";
            return;
        }

        clearAssemblers();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(binding.sigma), binding.domains);
        }
        matAsm_->assemble();
        vecAsm_->assemble();

        // Flatten voltageBindings_ to map for applyDirichletBC
        std::map<int, const VariableNode*> voltageBCs;
        for (const auto& binding : voltageBindings_) {
            for (int bid : binding.boundaryIds) {
                voltageBCs[bid] = binding.voltage;
            }
        }
        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, voltageBCs);
        matAsm_->finalize();
    }

} // namespace mpfem

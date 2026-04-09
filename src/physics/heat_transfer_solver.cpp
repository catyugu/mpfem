#include "heat_transfer_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    bool HeatTransferSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, Real initialTemperature)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));

        fieldValues.createField("T", fes_.get(), TensorShape::scalar(), initialTemperature);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
        solver_ = SolverFactory::create(*solverConfig_);

        LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";

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

    void HeatTransferSolver::setMassProperties(const std::set<int>& domains, const VariableNode* rhoCp)
    {
        MassBinding binding;
        binding.domains = domains;
        binding.thermalMass = rhoCp;
        massBindings_.push_back(std::move(binding));
    }

    void HeatTransferSolver::addTemperatureBC(const std::set<int>& boundaryIds, const VariableNode* temperature)
    {
        temperatureBindings_.push_back({boundaryIds, temperature});
    }

    void HeatTransferSolver::addConvectionBC(const std::set<int>& boundaryIds, const VariableNode* h, const VariableNode* Tinf)
    {
        convectionBindings_.push_back({boundaryIds, h, Tinf});
    }

    void HeatTransferSolver::buildStiffnessMatrix(SparseMatrix& K)
    {
        matAsm_->clear();
        matAsm_->clearIntegrators();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(
                std::make_unique<DiffusionIntegrator>(binding.conductivity),
                binding.domains);
        }

        for (const auto& binding : convectionBindings_) {
            for (int bid : binding.boundaryIds) {
                matAsm_->addBoundaryIntegrator(
                    std::make_unique<ConvectionMassIntegrator>(binding.h), bid);
            }
        }

        matAsm_->assemble();
        K = matAsm_->matrix();
    }

    void HeatTransferSolver::buildMassMatrix(SparseMatrix& M)
    {
        if (massBindings_.empty()) {
            M.resize(0, 0);
            return;
        }
        BilinearFormAssembler massAsm(fes_.get());
        for (const auto& binding : massBindings_) {
            massAsm.addDomainIntegrator(
                std::make_unique<MassIntegrator>(binding.thermalMass),
                binding.domains);
        }
        massAsm.assemble();
        M = massAsm.matrix();
    }

    void HeatTransferSolver::buildRHS(Vector& F)
    {
        vecAsm_->clear();
        vecAsm_->clearIntegrators();

        for (const auto& binding : convectionBindings_) {
            for (int bid : binding.boundaryIds) {
                vecAsm_->addBoundaryIntegrator(
                    std::make_unique<ConvectionLFIntegrator>(binding.h, binding.Tinf), bid);
            }
        }

        for (const auto& binding : heatSourceBindings_) {
            vecAsm_->addDomainIntegrator(
                std::make_unique<DomainLFIntegrator>(binding.source),
                binding.domains);
        }

        vecAsm_->assemble();
        F = vecAsm_->vector();
    }

    void HeatTransferSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution)
    {
        std::map<int, const VariableNode*> temperatureBCs;
        for (const auto& binding : temperatureBindings_) {
            for (int bid : binding.boundaryIds) {
                temperatureBCs[bid] = binding.temperature;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, temperatureBCs);
    }

    bool HeatTransferSolver::solveLinearSystem(SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!solver_) {
            LOG_ERROR << "HeatTransferSolver: solver not available";
            return false;
        }
        Vector rhs = b;
        applyEssentialBCs(A, rhs, x);
        A.makeCompressed();
        solver_->setup(&A);
        solver_->apply(rhs, x);
        field().markUpdated();
        LOG_INFO << fieldName() << " linear solve converged in " << iterations() << " iterations";
        return true;
    }

} // namespace mpfem
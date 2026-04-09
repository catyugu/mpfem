#include "heat_transfer_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

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

    void HeatTransferSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix)
    {
        std::map<int, const VariableNode*> temperatureBCs;
        for (const auto& binding : temperatureBindings_) {
            for (int bid : binding.boundaryIds) {
                temperatureBCs[bid] = binding.temperature;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, temperatureBCs, updateMatrix);
    }

    std::uint64_t HeatTransferSolver::getMatrixRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : conductivityBindings_) {
            if (b.conductivity)
                rev = std::max(rev, b.conductivity->revision());
        }
        for (const auto& b : convectionBindings_) {
            if (b.h)
                rev = std::max(rev, b.h->revision());
        }
        return rev;
    }

    std::uint64_t HeatTransferSolver::getMassRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : massBindings_) {
            if (b.thermalMass)
                rev = std::max(rev, b.thermalMass->revision());
        }
        return rev;
    }

    std::uint64_t HeatTransferSolver::getRhsRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : heatSourceBindings_) {
            if (b.source)
                rev = std::max(rev, b.source->revision());
        }
        for (const auto& b : convectionBindings_) {
            if (b.h)
                rev = std::max(rev, b.h->revision());
            if (b.Tinf)
                rev = std::max(rev, b.Tinf->revision());
        }
        return rev;
    }

    std::uint64_t HeatTransferSolver::getBcRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : temperatureBindings_) {
            if (b.temperature)
                rev = std::max(rev, b.temperature->revision());
        }
        return rev;
    }

} // namespace mpfem
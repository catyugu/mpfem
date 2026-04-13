#include "electrostatics_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    void ElectrostaticsSolver::setElectricalConductivity(const std::set<int>& domains, const VariableNode* sigma)
    {
        conductivityBindings_.push_back({domains, sigma});
    }

    void ElectrostaticsSolver::addVoltageBC(const std::set<int>& boundaryIds, const VariableNode* voltage)
    {
        voltageBindings_.push_back({boundaryIds, voltage});
    }

    void ElectrostaticsSolver::buildStiffnessMatrix(SparseMatrix& K)
    {
        matAsm_->clear();
        matAsm_->clearIntegrators();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(
                std::make_unique<DiffusionIntegrator>(binding.sigma),
                binding.domains);
        }
        matAsm_->assemble();
        K = matAsm_->matrix();
    }

    void ElectrostaticsSolver::buildRHS(Vector& F)
    {
        // Electrostatics typically has no volume source, but keep the pattern
        vecAsm_->clear();
        vecAsm_->assemble();
        F = vecAsm_->vector();
    }

    void ElectrostaticsSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix)
    {
        std::map<int, const VariableNode*> voltageBCs;
        for (const auto& binding : voltageBindings_) {
            for (int bid : binding.boundaryIds) {
                voltageBCs[bid] = binding.voltage;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, voltageBCs, updateMatrix);
    }

    std::uint64_t ElectrostaticsSolver::getMatrixRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : conductivityBindings_) {
            if (b.sigma)
                rev = std::max(rev, b.sigma->revision());
        }
        return rev;
    }

    std::uint64_t ElectrostaticsSolver::getRhsRevision() const
    {
        return 0; // Electrostatics typically has no volume source
    }

    std::uint64_t ElectrostaticsSolver::getBcRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : voltageBindings_) {
            if (b.voltage)
                rev = std::max(rev, b.voltage->revision());
        }
        return rev;
    }

} // namespace mpfem
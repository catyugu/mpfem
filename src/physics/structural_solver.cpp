#include "structural_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    void StructuralSolver::addElasticity(const std::set<int>& domains,
        const VariableNode* E,
        const VariableNode* nu)
    {
        elasticityBindings_.push_back({domains, E, nu});
    }

    void StructuralSolver::setStrainLoad(const std::set<int>& domains, const VariableNode* stress)
    {
        strainLoadBindings_.push_back({domains, stress});
    }

    void StructuralSolver::addFixedDisplacementBC(const std::set<int>& boundaryIds, const VariableNode* displacement)
    {
        displacementBindings_.push_back({boundaryIds, displacement});
    }

    void StructuralSolver::buildStiffnessMatrix(SparseMatrix& K)
    {
        matAsm_->clear();
        matAsm_->clearIntegrators();

        for (const auto& binding : elasticityBindings_) {
            matAsm_->addDomainIntegrator(
                std::make_unique<ElasticityIntegrator>(binding.E, binding.nu, fes_->vdim()),
                binding.domains);
        }
        matAsm_->assemble();
        K = matAsm_->matrix();
    }

    void StructuralSolver::buildRHS(Vector& F)
    {
        vecAsm_->clear();
        vecAsm_->clearIntegrators();

        for (const auto& binding : strainLoadBindings_) {
            vecAsm_->addDomainIntegrator(
                std::make_unique<StrainLoadIntegrator>(binding.stress, fes_->vdim()),
                binding.domains);
        }
        vecAsm_->assemble();
        F = vecAsm_->vector();
    }

    void StructuralSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution, bool updateMatrix)
    {
        std::map<int, const VariableNode*> displacementBCs;
        for (const auto& binding : displacementBindings_) {
            for (int bid : binding.boundaryIds) {
                displacementBCs[bid] = binding.displacement;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, displacementBCs, updateMatrix);
    }

    std::uint64_t StructuralSolver::getMatrixRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : elasticityBindings_) {
            if (b.E)
                rev = std::max(rev, b.E->revision());
            if (b.nu)
                rev = std::max(rev, b.nu->revision());
        }
        return rev;
    }

    std::uint64_t StructuralSolver::getRhsRevision() const
    {
        std::uint64_t rev = 0;
        for (const auto& b : strainLoadBindings_) {
            if (b.stress)
                rev = std::max(rev, b.stress->revision());
        }
        return rev;
    }

} // namespace mpfem
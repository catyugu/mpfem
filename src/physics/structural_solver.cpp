#include "structural_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    bool StructuralSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, Real initialDisplacement)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec), 3);

        fieldValues.createField("u", fes_.get(), TensorShape::vector(3));

        // Set initial displacement value for all components
        fieldValues.current("u").values().setConstant(initialDisplacement);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
        solver_ = SolverFactory::create(*solverConfig_);

        LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
        return true;
    }

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

    void StructuralSolver::addFixedDisplacementBC(const std::set<int>& boundaryIds, const Vector3& displacement)
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

    void StructuralSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution)
    {
        std::map<int, Vector3> displacementBCs;
        for (const auto& binding : displacementBindings_) {
            for (int bid : binding.boundaryIds) {
                displacementBCs[bid] = binding.displacement;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, displacementBCs, 3);
    }

} // namespace mpfem
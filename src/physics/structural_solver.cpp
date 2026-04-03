#include "structural_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"

namespace mpfem {

    bool StructuralSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialDisplacement)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec), 3);

        fieldValues.createVectorField(FieldId::Displacement, fes_.get(), 3);

        // Set initial displacement value for all components
        fieldValues.current(FieldId::Displacement).values().setConstant(initialDisplacement);

        matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
        vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());

        LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
        return true;
    }

    void StructuralSolver::addElasticity(const std::set<int>& domains,
        const Coefficient* E,
        const Coefficient* nu)
    {
        elasticityBindings_.push_back({domains, E, nu});
    }

    void StructuralSolver::setStrainLoad(const std::set<int>& domains, const MatrixCoefficient* stress)
    {
        strainLoadBindings_.push_back({domains, stress});
    }

    void StructuralSolver::addFixedDisplacementBC(const std::set<int>& boundaryIds, const VectorCoefficient* displacement)
    {
        for (int bid : boundaryIds)
            displacementBCs_[bid] = displacement;
    }

    void StructuralSolver::assemble()
    {
        ScopedTimer timer("Structural assemble");

        if (!fes_)
            return;

        if (elasticityBindings_.empty()) {
            LOG_ERROR << "StructuralSolver: material not set";
            return;
        }

        const std::uint64_t currentStiffnessTag = stateTagOfRange(elasticityBindings_);
        const std::uint64_t currentLoadTag = stateTagOfRange(strainLoadBindings_);
        const std::uint64_t currentBcTag = stateTagOfRange(displacementBCs_);

        const bool rebuildStiffness = stiffnessAssemblyState_.needsRebuild(currentStiffnessTag);
        const bool rebuildLoad = loadAssemblyState_.needsRebuild(currentLoadTag);
        const bool bcChanged = bcAssemblyState_.needsRebuild(currentBcTag);

        if (!rebuildStiffness && !rebuildLoad && !bcChanged) {
            LOG_DEBUG << "Structural assemble skipped (coefficients unchanged)";
            return;
        }

        if (rebuildStiffness) {
            matAsm_->clear();
            matAsm_->clearIntegrators();

            for (const auto& binding : elasticityBindings_) {
                matAsm_->addDomainIntegrator(
                    std::make_unique<ElasticityIntegrator>(binding.E, binding.nu, fes_->vdim()),
                    binding.domains);
            }
            matAsm_->assemble();
            stiffnessMatrixBeforeBC_ = matAsm_->matrix();
        }
        // Skip copy-back when rebuildStiffness=false - applyDirichletBC will modify in-place anyway

        if (rebuildLoad) {
            vecAsm_->clear();
            vecAsm_->clearIntegrators();

            for (const auto& binding : strainLoadBindings_) {
                vecAsm_->addDomainIntegrator(std::make_unique<StrainLoadIntegrator>(
                                                 binding.stress, fes_->vdim()),
                    binding.domains);
            }
            vecAsm_->assemble();
            rhsBeforeBC_ = vecAsm_->vector();
        }
        else {
            vecAsm_->vector() = rhsBeforeBC_;
        }

        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, displacementBCs_, 3);
        matAsm_->finalize();

        // Mark matrix as updated so solver will re-setup on next solve()
        matrix_needs_update_ = true;

        stiffnessAssemblyState_.update(currentStiffnessTag);
        loadAssemblyState_.update(currentLoadTag);
        bcAssemblyState_.update(currentBcTag);
    }

} // namespace mpfem

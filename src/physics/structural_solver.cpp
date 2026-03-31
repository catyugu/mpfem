#include "structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem
{

    bool StructuralSolver::initialize(const Mesh &mesh, FieldValues &fieldValues, int order, double initialDisplacement)
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
        solver_ = SolverFactory::create(solverConfig_);

        LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
        return true;
    }

    void StructuralSolver::setYoungModulus(const std::set<int> &domains, const Coefficient *E)
    {
        youngModulus_.set(domains, E);
    }

    void StructuralSolver::setPoissonRatio(const std::set<int> &domains, const Coefficient *nu)
    {
        poissonRatio_.set(domains, nu);
    }

    void StructuralSolver::setStrainLoad(const std::set<int> &domains, const MatrixCoefficient *stress)
    {
        strainLoad_.set(domains, stress);
    }

    void StructuralSolver::addFixedDisplacementBC(const std::set<int> &boundaryIds, const VectorCoefficient *displacement)
    {
        for (int bid : boundaryIds)
            displacementBCs_[bid] = displacement;
    }

    void StructuralSolver::assemble()
    {
        ScopedTimer timer("Structural assemble");

        if (!fes_)
            return;

        if (youngModulus_.empty() || poissonRatio_.empty())
        {
            LOG_ERROR << "StructuralSolver: material not set";
            return;
        }

        clearAssemblers();

        matAsm_->addDomainIntegrator(std::make_unique<ElasticityIntegrator>(&youngModulus_, &poissonRatio_, fes_->vdim()));
        matAsm_->assemble();

        if (!strainLoad_.empty())
        {
            vecAsm_->addDomainIntegrator(std::make_unique<StrainLoadIntegrator>(
                &strainLoad_, fes_->vdim()));
        }
        vecAsm_->assemble();

        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, displacementBCs_, 3);
        matAsm_->finalize();
    }

} // namespace mpfem

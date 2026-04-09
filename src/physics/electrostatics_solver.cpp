#include "electrostatics_solver.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "solver/solver_factory.hpp"

namespace mpfem {

    bool ElectrostaticsSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, Real initialPotential)
    {
        mesh_ = &mesh;
        fieldValues_ = &fieldValues;
        order_ = order;

        auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
        fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));

        fieldValues.createField("V", fes_.get(), TensorShape::scalar(), initialPotential);

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

    void ElectrostaticsSolver::applyEssentialBCs(SparseMatrix& A, Vector& rhs, Vector& solution)
    {
        std::map<int, const VariableNode*> voltageBCs;
        for (const auto& binding : voltageBindings_) {
            for (int bid : binding.boundaryIds) {
                voltageBCs[bid] = binding.voltage;
            }
        }
        applyDirichletBC(A, rhs, solution, *fes_, *mesh_, voltageBCs);
    }

    bool ElectrostaticsSolver::solveLinearSystem(SparseMatrix& A, Vector& x, const Vector& b)
    {
        if (!solver_) {
            LOG_ERROR << "ElectrostaticsSolver: solver not available";
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
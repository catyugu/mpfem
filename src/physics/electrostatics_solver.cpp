#include "electrostatics_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool ElectrostaticsSolver::initialize(const Mesh& mesh, FieldValues& fieldValues) {
    mesh_ = &mesh;
    fieldValues_ = &fieldValues;
    
    // Create finite element space (FESpace owns FECollection)
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    // Register electric potential field with FieldValues
    fieldValues.createScalarField(FieldId::ElectricPotential, fes_.get(), 0.0);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    
    createSolver();
    
    LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void ElectrostaticsSolver::setConductivity(const std::set<int>& domains, const Coefficient* sigma) {
    conductivity_.set(domains, sigma);
}

void ElectrostaticsSolver::addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage) {
    for (int bid : boundaryIds) {
        voltageBCs_[bid] = voltage;
    }
}

void ElectrostaticsSolver::assemble() {
    ScopedTimer timer("Electrostatics assemble");
    
    if (conductivity_.empty()) {
        LOG_ERROR << "ElectrostaticsSolver: conductivity not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    auto integ = std::make_unique<DiffusionIntegrator>(&conductivity_);
    matAsm_->addDomainIntegrator(std::move(integ));
    matAsm_->assemble();
    vecAsm_->assemble();
    
    // Apply boundary conditions
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(),
                     *fes_, *mesh_, voltageBCs_);
    matAsm_->finalize();
}

bool ElectrostaticsSolver::solve() {
    if (!solver_) return false;
    bool ok = solver_->solve(matAsm_->matrix(), field().values(), vecAsm_->vector());
    if (ok) {
        LOG_INFO << "Electrostatics solver converged!";
    }
    return ok;
}

} // namespace mpfem
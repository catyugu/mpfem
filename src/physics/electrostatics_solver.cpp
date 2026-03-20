#include "electrostatics_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool ElectrostaticsSolver::initialize(const Mesh& mesh, FieldValues& fieldValues) {
    mesh_ = &mesh;
    fieldValues_ = &fieldValues;
    
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    fieldValues.createScalarField(FieldId::ElectricPotential, fes_.get(), 0.0);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    createSolver();
    
    LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void ElectrostaticsSolver::setConductivity(const std::set<int>& domains, const MatrixCoefficient* sigma) {
    conductivity_.set(domains, sigma);
}

void ElectrostaticsSolver::addVoltageBC(const std::set<int>& boundaryIds, const Coefficient* voltage) {
    for (int bid : boundaryIds) voltageBCs_[bid] = voltage;
}

void ElectrostaticsSolver::assemble() {
    ScopedTimer timer("Electrostatics assemble");
    
    if (conductivity_.empty()) {
        LOG_ERROR << "ElectrostaticsSolver: conductivity not set";
        return;
    }
    
    clearAssemblers();
    
    matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&conductivity_));
    matAsm_->assemble();
    vecAsm_->assemble();
    
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, voltageBCs_);
    matAsm_->finalize();
}

} // namespace mpfem
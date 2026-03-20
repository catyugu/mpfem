#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order) {
    mesh_ = &mesh;
    fieldValues_ = &fieldValues;
    order_ = order;
    
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    fieldValues.createScalarField(FieldId::Temperature, fes_.get(), 293.15);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    solver_ = SolverFactory::create(solverConfig_);
    
    LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void HeatTransferSolver::setConductivity(const std::set<int>& domains, const MatrixCoefficient* k) {
    conductivity_.set(domains, k);
}

void HeatTransferSolver::setHeatSource(const std::set<int>& domains, const Coefficient* Q) {
    heatSource_.set(domains, Q);
}

void HeatTransferSolver::addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature) {
    for (int bid : boundaryIds) temperatureBCs_[bid] = temperature;
}

void HeatTransferSolver::addConvectionBC(const std::set<int>& boundaryIds, const Coefficient* h, const Coefficient* Tinf) {
    for (int bid : boundaryIds) convBCs_[bid] = {h, Tinf};
}

void HeatTransferSolver::assemble() {
    ScopedTimer timer("HeatTransfer assemble");
    
    if (conductivity_.empty()) {
        LOG_ERROR << "HeatTransferSolver: conductivity not set";
        return;
    }
    
    clearAssemblers();
    
    matAsm_->addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&conductivity_));
    
    for (const auto& [bid, bc] : convBCs_) {
        matAsm_->addBoundaryIntegrator(std::make_unique<ConvectionMassIntegrator>(bc.h), bid);
        vecAsm_->addBoundaryIntegrator(std::make_unique<ConvectionLFIntegrator>(bc.h, bc.Tinf), bid);
    }
    
    matAsm_->assemble();
    
    if (!heatSource_.empty()) {
        vecAsm_->addDomainIntegrator(std::make_unique<DomainLFIntegrator>(&heatSource_));
    }
    vecAsm_->assemble();
    
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, temperatureBCs_);
    matAsm_->finalize();
}

}  // namespace mpfem

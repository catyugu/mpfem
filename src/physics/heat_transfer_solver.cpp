#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh, FieldValues& fieldValues) {
    mesh_ = &mesh;
    fieldValues_ = &fieldValues;
    
    // Create finite element space (FESpace owns FECollection)
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    // Register temperature field with FieldValues
    fieldValues.createScalarField(FieldId::Temperature, fes_.get(), 293.15);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    
    createSolver();
    
    LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void HeatTransferSolver::setConductivity(const std::set<int>& domains, const Coefficient* k) {
    conductivity_.set(domains, k);
}

void HeatTransferSolver::setHeatSource(const std::set<int>& domains, const Coefficient* Q) {
    heatSource_.set(domains, Q);
}

void HeatTransferSolver::addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature) {
    for (int bid : boundaryIds) {
        temperatureBCs_[bid] = temperature;
    }
}

void HeatTransferSolver::addConvectionBC(const std::set<int>& boundaryIds, 
                                          const Coefficient* h, 
                                          const Coefficient* Tinf) {
    for (int bid : boundaryIds) {
        convBCs_[bid] = {h, Tinf};
    }
}

void HeatTransferSolver::assemble() {
    ScopedTimer timer("HeatTransfer assemble");
    
    if (conductivity_.empty()) {
        LOG_ERROR << "HeatTransferSolver: conductivity not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    // Diffusion integrator
    auto diff = std::make_unique<DiffusionIntegrator>(&conductivity_);
    matAsm_->addDomainIntegrator(std::move(diff));
    
    // Convection boundary conditions
    for (const auto& [bid, bc] : convBCs_) {
        // Convection BC: h(T - Tinf) = 0
        // Weak form: ∫ h T φ dΓ - ∫ h Tinf φ dΓ = 0
        // Matrix part: ∫ h φ_i φ_j dΓ
        auto convMat = std::make_unique<ConvectionMassIntegrator>(bc.h);
        matAsm_->addBoundaryIntegrator(std::move(convMat), bid);
        
        // Vector part: ∫ h Tinf φ_i dΓ
        auto convVec = std::make_unique<ConvectionLFIntegrator>(bc.h, bc.Tinf);
        vecAsm_->addBoundaryIntegrator(std::move(convVec), bid);
    }
    
    matAsm_->assemble();
    
    // Heat source
    if (!heatSource_.empty()) {
        auto src = std::make_unique<DomainLFIntegrator>(&heatSource_);
        vecAsm_->addDomainIntegrator(std::move(src));
    }
    
    vecAsm_->assemble();
    
    // Apply temperature boundary conditions
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(),
                     *fes_, *mesh_, temperatureBCs_);
    matAsm_->finalize();
}

bool HeatTransferSolver::solve() {
    if (!solver_) return false;
    bool ok = solver_->solve(matAsm_->matrix(), field().values(), vecAsm_->vector());
    if (ok) {
        LOG_INFO << "HeatTransfer converged!";
    }
    return ok;
}

}  // namespace mpfem
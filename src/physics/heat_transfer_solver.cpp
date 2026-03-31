#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh, FieldValues& fieldValues, int order, double initialTemperature) {
    mesh_ = &mesh;
    fieldValues_ = &fieldValues;
    order_ = order;
    
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    fieldValues.createScalarField(FieldId::Temperature, fes_.get(), initialTemperature);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    solver_ = SolverFactory::create(solverConfig_);
    
    LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";
    
    // Assemble mass matrix if density and specific heat are set
    if (!massBindings_.empty()) {
        assembleMassMatrix();
    }
    
    return true;
}

void HeatTransferSolver::setThermalConductivity(const std::set<int>& domains, const MatrixCoefficient* k) {
    conductivityBindings_.push_back({domains, k});
}

void HeatTransferSolver::setHeatSource(const std::set<int>& domains, const Coefficient* Q) {
    heatSourceBindings_.push_back({domains, Q});
}

void HeatTransferSolver::setMassProperties(const std::set<int>& domains,
                                           const Coefficient* rho,
                                           const Coefficient* Cp) {
    MassBinding binding;
    binding.domains = domains;
    binding.density = rho;
    binding.specificHeat = Cp;
    binding.rhoCp = std::make_unique<ProductCoefficient>(rho, Cp);
    massBindings_.push_back(std::move(binding));
    massMatrixAssembled_ = false;
}

void HeatTransferSolver::assembleMassMatrix() {
    if (massBindings_.empty()) {
        LOG_ERROR << "HeatTransferSolver: mass properties not set for mass matrix";
        return;
    }

    auto massAsm = std::make_unique<BilinearFormAssembler>(fes_.get());
    for (const auto& binding : massBindings_) {
        massAsm->addDomainIntegrator(
            std::make_unique<MassIntegrator>(binding.rhoCp.get()),
            binding.domains);
    }
    massAsm->assemble();
    
    massMatrix_ = std::move(massAsm->matrix());
    massMatrixAssembled_ = true;
    
    LOG_INFO << "HeatTransferSolver: mass matrix assembled (" << massMatrix_.rows() << "x" << massMatrix_.cols() << ")";
}

void HeatTransferSolver::addTemperatureBC(const std::set<int>& boundaryIds, const Coefficient* temperature) {
    for (int bid : boundaryIds) temperatureBCs_[bid] = temperature;
}

void HeatTransferSolver::addConvectionBC(const std::set<int>& boundaryIds, const Coefficient* h, const Coefficient* Tinf) {
    for (int bid : boundaryIds) convBCs_[bid] = {h, Tinf};
}

void HeatTransferSolver::assemble() {
    ScopedTimer timer("HeatTransfer assemble");
    
    if (conductivityBindings_.empty()) {
        LOG_ERROR << "HeatTransferSolver: conductivity not set";
        return;
    }
    
    // Lazy mass matrix assembly - assemble if needed but not yet assembled
    if (!massMatrixAssembled_ && !massBindings_.empty()) {
        assembleMassMatrix();
    }
    
    clearAssemblers();
    
    for (const auto& binding : conductivityBindings_) {
        matAsm_->addDomainIntegrator(
            std::make_unique<DiffusionIntegrator>(binding.conductivity),
            binding.domains);
    }
    
    for (const auto& [bid, bc] : convBCs_) {
        matAsm_->addBoundaryIntegrator(std::make_unique<ConvectionMassIntegrator>(bc.h), bid);
        vecAsm_->addBoundaryIntegrator(std::make_unique<ConvectionLFIntegrator>(bc.h, bc.Tinf), bid);
    }
    
    matAsm_->assemble();
    
    // Cache stiffness matrix before BC application (needed for transient time integrators like BDF1)
    stiffnessMatrixBeforeBC_ = matAsm_->matrix();
    
    for (const auto& binding : heatSourceBindings_) {
        vecAsm_->addDomainIntegrator(
            std::make_unique<DomainLFIntegrator>(binding.source),
            binding.domains);
    }
    vecAsm_->assemble();
    
    // Cache RHS before BC application (needed for transient time integrators like BDF1)
    rhsBeforeBC_ = vecAsm_->vector();
    
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, temperatureBCs_);
    matAsm_->finalize();
}

bool HeatTransferSolver::solveLinearSystem(const SparseMatrix& A, Vector& x, const Vector& b) {
    if (!solver_) {
        LOG_ERROR << "HeatTransferSolver: solver not available";
        return false;
    }
    
    // Make a copy since applyDirichletBC modifies the matrix
    SparseMatrix A_copy = A;
    Vector b_copy = b;
    
    // Apply boundary conditions to the combined system
    applyDirichletBC(A_copy, b_copy, x, *fes_, *mesh_, temperatureBCs_);
    A_copy.makeCompressed();
    
    // Solve the system
    bool ok = solver_->solve(A_copy, x, b_copy);
    
    if (ok) {
        LOG_INFO << fieldName() << " linear solve converged in " << iterations() << " iterations";
    } else {
        LOG_ERROR << fieldName() << " linear solve failed, residual: " << residual();
    }
    
    return ok;
}

}  // namespace mpfem

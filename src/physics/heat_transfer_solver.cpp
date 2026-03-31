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
    massBindings_.push_back(std::move(binding));
    massMatrixAssembled_ = false;
}

void HeatTransferSolver::assembleMassMatrix() {
    if (massBindings_.empty()) {
        LOG_ERROR << "HeatTransferSolver: mass properties not set for mass matrix";
        return;
    }

    auto massAsm = std::make_unique<BilinearFormAssembler>(fes_.get());
    std::vector<std::unique_ptr<FunctionCoefficient>> rhoCpCoefficients;
    rhoCpCoefficients.reserve(massBindings_.size());

    for (const auto& binding : massBindings_) {
        // Keep composite coefficients alive until matrix assembly completes.
        rhoCpCoefficients.push_back(std::make_unique<FunctionCoefficient>(
            [&binding](ElementTransform& trans, Real& result, Real t) {
                Real rho = 1.0, Cp = 1.0;
                if (binding.density) binding.density->eval(trans, rho, t);
                if (binding.specificHeat) binding.specificHeat->eval(trans, Cp, t);
                result = rho * Cp;
            },
            binding.stateTag()));
        
        massAsm->addDomainIntegrator(
            std::make_unique<MassIntegrator>(rhoCpCoefficients.back().get()),
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

    const std::uint64_t currentMassTag = stateTagOfRange(massBindings_);
    
    // Rebuild mass matrix only when rho/cp dependent coefficients changed.
    if (!massBindings_.empty() &&
        (!massMatrixAssembled_ || massAssemblyState_.needsRebuild(currentMassTag))) {
        assembleMassMatrix();
        massAssemblyState_.update(currentMassTag);
    }

    const std::uint64_t convectionStiffnessTag = stateTagOfRange(
        convBCs_,
        [](const auto& entry) {
            return combineTag(stateTagOf(entry.first), entry.second.stiffnessTag());
        });
    const std::uint64_t currentStiffnessTag = combineTag(
        stateTagOfRange(conductivityBindings_),
        convectionStiffnessTag);

    const std::uint64_t convectionLoadTag = stateTagOfRange(
        convBCs_,
        [](const auto& entry) {
            return combineTag(stateTagOf(entry.first), entry.second.loadTag());
        });
    const std::uint64_t currentLoadTag = combineTag(
        stateTagOfRange(heatSourceBindings_),
        convectionLoadTag);

    const std::uint64_t currentBcTag = stateTagOfRange(temperatureBCs_);

    const bool rebuildStiffness = stiffnessAssemblyState_.needsRebuild(currentStiffnessTag);
    const bool rebuildLoad = loadAssemblyState_.needsRebuild(currentLoadTag);
    const bool bcChanged = bcAssemblyState_.needsRebuild(currentBcTag);

    if (!rebuildStiffness && !rebuildLoad && !bcChanged) {
        LOG_DEBUG << "HeatTransfer assemble skipped (coefficients unchanged)";
        return;
    }
    
    if (rebuildStiffness) {
        matAsm_->clear();
        matAsm_->clearIntegrators();

        for (const auto& binding : conductivityBindings_) {
            matAsm_->addDomainIntegrator(
                std::make_unique<DiffusionIntegrator>(binding.conductivity),
                binding.domains);
        }

        for (const auto& [bid, bc] : convBCs_) {
            matAsm_->addBoundaryIntegrator(std::make_unique<ConvectionMassIntegrator>(bc.h), bid);
        }

        matAsm_->assemble();
        stiffnessMatrixBeforeBC_ = matAsm_->matrix();
    } else {
        matAsm_->matrix() = stiffnessMatrixBeforeBC_;
    }

    if (rebuildLoad) {
        vecAsm_->clear();
        vecAsm_->clearIntegrators();

        for (const auto& [bid, bc] : convBCs_) {
            vecAsm_->addBoundaryIntegrator(std::make_unique<ConvectionLFIntegrator>(bc.h, bc.Tinf), bid);
        }

        for (const auto& binding : heatSourceBindings_) {
            vecAsm_->addDomainIntegrator(
                std::make_unique<DomainLFIntegrator>(binding.source),
                binding.domains);
        }
        vecAsm_->assemble();
        rhsBeforeBC_ = vecAsm_->vector();
    } else {
        vecAsm_->vector() = rhsBeforeBC_;
    }
    
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), field().values(), *fes_, *mesh_, temperatureBCs_);
    matAsm_->finalize();

    stiffnessAssemblyState_.update(currentStiffnessTag);
    loadAssemblyState_.update(currentLoadTag);
    bcAssemblyState_.update(currentBcTag);
}

bool HeatTransferSolver::solveLinearSystem(const SparseMatrix& A, Vector& x, const Vector& b) {
    if (!solver_) {
        LOG_ERROR << "HeatTransferSolver: solver not available";
        return false;
    }
    
    // Reuse persistent buffers to avoid repeated allocations.
    systemMatrix_ = A;
    systemRhs_ = b;
    
    // Apply boundary conditions to the combined system
    applyDirichletBC(systemMatrix_, systemRhs_, x, *fes_, *mesh_, temperatureBCs_);
    systemMatrix_.makeCompressed();
    
    // Solve the system
    bool ok = solver_->solve(systemMatrix_, x, systemRhs_);
    
    if (ok) {
        field().markUpdated();
        LOG_INFO << fieldName() << " linear solve converged in " << iterations() << " iterations";
    } else {
        LOG_ERROR << fieldName() << " linear solve failed, residual: " << residual();
    }
    
    return ok;
}

}  // namespace mpfem

#include "structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool StructuralSolver::initialize(const Mesh& mesh,
                                   const PWConstCoefficient& youngModulus,
                                   const PWConstCoefficient& poissonRatio) {
    mesh_ = &mesh;
    EInternal_ = youngModulus;
    nuInternal_ = poissonRatio;
    
    // Create vector H1 elements for displacement (vdim=3)
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get(), 3);  // 3D displacement
    
    u_ = std::make_unique<GridFunction>(fes_.get());
    u_->setZero();
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    solver_ = SolverFactory::create(solverType_, maxIter_, tol_);
    
    LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void StructuralSolver::assemble() {
    if (!fes_) return;
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    // Add elasticity integrator
    auto elasticity = std::make_unique<ElasticityIntegrator>(&EInternal_, &nuInternal_);
    matAsm_->addDomainIntegrator(std::move(elasticity));
    matAsm_->assemble();
    
    // Add thermal strain if temperature field is set
    if (T_ && alphaT_) {
        auto thermalLoad = std::make_unique<ThermalLoadIntegrator>(
            &EInternal_, &nuInternal_, alphaT_, T_, Tref_);
        vecAsm_->addDomainIntegrator(std::move(thermalLoad));
    }
    vecAsm_->assemble();
    
    // Apply Dirichlet BCs (simplified: only supports fixing all components for now)
    // For structural problems, we typically fix boundaries entirely
    for (const auto& [bid, disp] : bcValues_) {
        // For vector field, apply BC to all components
        // This is a simplified approach
        Real val = disp.x();  // Use x-component as the value
        std::map<int, Real> scalarBC;
        scalarBC[bid] = val;
        applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), u_->values(),
                         *fes_, *mesh_, scalarBC);
    }
    
    matAsm_->finalize();
}

bool StructuralSolver::solve() {
    if (!matAsm_ || !vecAsm_) return false;
    
    bool success = solver_->solve(matAsm_->matrix(), u_->values(), vecAsm_->vector());
    
    if (success) {
        iter_ = solver_->iterations();
        res_ = solver_->residual();
        LOG_INFO << "StructuralSolver: displacement norm = " << u_->values().norm();
        // computeStressStrain();
    }
    
    return success;
}

void StructuralSolver::computeStressStrain() {
    // TODO: Implement stress/strain post-processing
    // This requires evaluating the strain from displacement gradient
    // and computing stress from constitutive relation
}

}  // namespace mpfem
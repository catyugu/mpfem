#include "electrostatics_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool ElectrostaticsSolver::initialize(const Mesh& mesh) {
    mesh_ = &mesh;
    
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get());
    V_ = std::make_unique<GridFunction>(fes_.get());
    V_->setZero();
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    createSolver();
    
    LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void ElectrostaticsSolver::assemble() {
    ScopedTimer timer("Electrostatics assemble");
    
    if (!sigma_) {
        LOG_ERROR << "ElectrostaticsSolver: conductivity not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    auto integ = std::make_unique<DiffusionIntegrator>(sigma_);
    matAsm_->addDomainIntegrator(std::move(integ));
    matAsm_->assemble();
    vecAsm_->assemble();
    
    // 应用边界条件
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), V_->values(),
                     *fes_, *mesh_, voltageBCs_);
    matAsm_->finalize();
}

bool ElectrostaticsSolver::solve() {
    if (!solver_) return false;
    bool ok = solver_->solve(matAsm_->matrix(), V_->values(), vecAsm_->vector());
    if (ok) {
        iter_ = solver_->iterations();
        res_ = solver_->residual();
        LOG_INFO << "Electrostatics converged: iter=" << iter_ << " res=" << res_;
    }
    return ok;
}

} // namespace mpfem

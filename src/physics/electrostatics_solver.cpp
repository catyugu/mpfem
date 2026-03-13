#include "electrostatics_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

// =============================================================================
// TempDepSigmaCoefficient
// =============================================================================

Real TempDepSigmaCoefficient::eval(ElementTransform& trans) const {
    int attr = static_cast<int>(trans.attribute());
    if (attr < 1 || attr > static_cast<int>(rho0_.size())) return 1.0;
    
    Real rho0 = rho0_[attr - 1];
    
    // rho0 = 0 或负数表示使用常量电导率
    if (rho0 <= 0.0) {
        return sigma0_[attr - 1];
    }
    
    Real alpha = alpha_[attr - 1];
    Real tref = tref_[attr - 1];
    
    // 获取温度
    Real temp = tref;
    if (T_) {
        const auto& ip = trans.integrationPoint();
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        temp = T_->eval(trans.elementIndex(), xi);
    }
    
    // 线性电阻率模型: rho = rho0 * (1 + alpha * (T - Tref))
    Real rho = rho0 * (1.0 + alpha * (temp - tref));
    return 1.0 / rho;
}

// =============================================================================
// ElectrostaticsSolver
// =============================================================================

bool ElectrostaticsSolver::initialize(const Mesh& mesh, 
                                       const PWConstCoefficient& conductivity) {
    mesh_ = &mesh;
    
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get());
    V_ = std::make_unique<GridFunction>(fes_.get());
    V_->setZero();
    
    sigmaInternal_ = conductivity;
    sigma_ = &sigmaInternal_;
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    solver_ = SolverFactory::create(solverType_, maxIter_, tol_);
    
    LOG_INFO << "ElectrostaticsSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void ElectrostaticsSolver::assemble() {
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
                     *fes_, *mesh_, bcValues_);
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

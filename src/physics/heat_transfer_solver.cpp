#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh, const Coefficient& conductivity) {
    mesh_ = &mesh;
    k_ = &conductivity;
    
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get());
    T_ = std::make_unique<GridFunction>(fes_.get());
    T_->values().setConstant(293.15);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    solver_ = SolverFactory::create(solverType_, maxIter_, tol_);
    
    LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void HeatTransferSolver::assemble() {
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    auto diff = std::make_unique<DiffusionIntegrator>(k_);
    matAsm_->addDomainIntegrator(std::move(diff));
    
    for (const auto& [bid, bc] : convBCs_) {
        // 对流边界条件: h(T - Tinf) = 0
        // 弱形式: ∫ h T φ dΓ - ∫ h Tinf φ dΓ = 0
        // 矩阵部分: ∫ h φ_i φ_j dΓ
        // 向量部分: ∫ h Tinf φ_i dΓ
        
        auto convMat = std::make_unique<ConvectionMassIntegrator>(
            std::make_unique<ConstantCoefficient>(bc.h));
        matAsm_->addBoundaryIntegrator(std::move(convMat), bid);
        
        auto convVec = std::make_unique<ConvectionLFIntegrator>(
            std::make_unique<ConstantCoefficient>(bc.h),
            std::make_unique<ConstantCoefficient>(bc.Tinf));
        vecAsm_->addBoundaryIntegrator(std::move(convVec), bid);
    }
    
    matAsm_->assemble();
    
    if (heatSource_) {
        auto src = std::make_unique<DomainLFIntegrator>(heatSource_);
        vecAsm_->addDomainIntegrator(std::move(src));
    }
    
    vecAsm_->assemble();
    
    // 应用Dirichlet边界条件
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), T_->values(),
                     *fes_, *mesh_, bcValues_);
    matAsm_->finalize();
}

bool HeatTransferSolver::solve() {
    if (!solver_) return false;
    bool ok = solver_->solve(matAsm_->matrix(), T_->values(), vecAsm_->vector());
    if (ok) {
        iter_ = solver_->iterations();
        res_ = solver_->residual();
        LOG_INFO << "HeatTransfer converged: iter=" << iter_ << " res=" << res_;
    }
    return ok;
}

}  // namespace mpfem
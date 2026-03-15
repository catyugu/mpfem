#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh) {
    mesh_ = &mesh;
    
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
    ScopedTimer timer("HeatTransfer assemble");
    
    if (!k_) {
        LOG_ERROR << "HeatTransferSolver: conductivity not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    // 清除之前持有的边界条件系数
    ownedConvH_.clear();
    ownedConvTinf_.clear();
    
    // 扩散积分器
    auto diff = std::make_unique<DiffusionIntegrator>(k_);
    matAsm_->addDomainIntegrator(std::move(diff));
    
    // 对流边界条件
    for (const auto& [bid, bc] : convBCs_) {
        // 创建并持有系数（避免悬空指针）
        auto hCoef = std::make_unique<ConstantCoefficient>(bc.h);
        auto tinfCoef = std::make_unique<ConstantCoefficient>(bc.Tinf);
        
        const Coefficient* hPtr = hCoef.get();
        const Coefficient* tinfPtr = tinfCoef.get();
        
        ownedConvH_.push_back(std::move(hCoef));
        ownedConvTinf_.push_back(std::move(tinfCoef));
        
        // 对流边界条件: h(T - Tinf) = 0
        // 弱形式: ∫ h T φ dΓ - ∫ h Tinf φ dΓ = 0
        // 矩阵部分: ∫ h φ_i φ_j dΓ
        auto convMat = std::make_unique<ConvectionMassIntegrator>(hPtr);
        matAsm_->addBoundaryIntegrator(std::move(convMat), bid);
        
        // 向量部分: ∫ h Tinf φ_i dΓ
        auto convVec = std::make_unique<ConvectionLFIntegrator>(hPtr, tinfPtr);
        vecAsm_->addBoundaryIntegrator(std::move(convVec), bid);
    }
    
    matAsm_->assemble();
    
    // 热源
    if (heatSource_) {
        auto src = std::make_unique<DomainLFIntegrator>(heatSource_);
        vecAsm_->addDomainIntegrator(std::move(src));
    }
    
    vecAsm_->assemble();
    
    // 应用温度边界条件
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), T_->values(),
                     *fes_, *mesh_, temperatureBCs_);
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

#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh) {
    mesh_ = &mesh;
    
    // 创建有限元空间（FESpace 拥有 FECollection）
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec));
    
    T_ = std::make_unique<GridFunction>(fes_.get());
    T_->values().setConstant(293.15);
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
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
    
    // 扩散积分器
    auto diff = std::make_unique<DiffusionIntegrator>(&conductivity_);
    matAsm_->addDomainIntegrator(std::move(diff));
    
    // 对流边界条件
    for (const auto& [bid, bc] : convBCs_) {
        // 对流边界条件: h(T - Tinf) = 0
        // 弱形式: ∫ h T φ dΓ - ∫ h Tinf φ dΓ = 0
        // 矩阵部分: ∫ h φ_i φ_j dΓ
        auto convMat = std::make_unique<ConvectionMassIntegrator>(bc.h);
        matAsm_->addBoundaryIntegrator(std::move(convMat), bid);
        
        // 向量部分: ∫ h Tinf φ_i dΓ
        auto convVec = std::make_unique<ConvectionLFIntegrator>(bc.h, bc.Tinf);
        vecAsm_->addBoundaryIntegrator(std::move(convVec), bid);
    }
    
    matAsm_->assemble();
    
    // 热源
    if (!heatSource_.empty()) {
        auto src = std::make_unique<DomainLFIntegrator>(&heatSource_);
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
        LOG_INFO << "HeatTransfer converged!";
    }
    return ok;
}

}  // namespace mpfem

#include "structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool StructuralSolver::initialize(const Mesh& mesh) {
    mesh_ = &mesh;
    
    // 创建向量H1单元 (vdim=3)，FESpace 拥有 FECollection
    auto fec = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, std::move(fec), 3);  // 3D位移
    
    u_ = std::make_unique<GridFunction>(fes_.get());
    u_->setZero();
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    
    createSolver();
    
    LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void StructuralSolver::setYoungModulus(const std::set<int>& domains, const Coefficient* E) {
    youngModulus_.set(domains, E);
}

void StructuralSolver::setPoissonRatio(const std::set<int>& domains, const Coefficient* nu) {
    poissonRatio_.set(domains, nu);
}

void StructuralSolver::setThermalExpansion(const std::set<int>& domains, const Coefficient* alphaT) {
    thermalExpansion_.set(domains, alphaT);
}

void StructuralSolver::addFixedDisplacementBC(const std::set<int>& boundaryIds, 
                                               const VectorCoefficient* displacement) {
    for (int bid : boundaryIds) {
        displacementBCs_[bid] = displacement;
    }
}

void StructuralSolver::assemble() {
    ScopedTimer timer("Structural assemble");
    
    if (!fes_) return;
    
    if (youngModulus_.empty() || poissonRatio_.empty()) {
        LOG_ERROR << "StructuralSolver: material not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    // 添加弹性积分器（向量场）
    auto elasticity = std::make_unique<ElasticityIntegrator>(&youngModulus_, &poissonRatio_);
    matAsm_->addDomainIntegrator(std::move(elasticity));
    matAsm_->assemble();
    
    // 添加热膨胀载荷积分器（如果有热膨胀系数）
    if (!thermalExpansion_.empty()) {
        auto thermalLoad = std::make_unique<ThermalLoadIntegrator>(
            &youngModulus_, &poissonRatio_, &thermalExpansion_);
        vecAsm_->addDomainIntegrator(std::move(thermalLoad));
    }
    
    vecAsm_->assemble();
    
    // 应用位移边界条件
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), u_->values(),
                     *fes_, *mesh_, displacementBCs_, 3);
    
    matAsm_->finalize();
}

bool StructuralSolver::solve() {
    if (!matAsm_ || !vecAsm_) return false;
    
    bool success = solver_->solve(matAsm_->matrix(), u_->values(), vecAsm_->vector());
    
    if (success) {
        LOG_INFO << "StructuralSolver: displacement norm = " << u_->values().norm();
    }
    
    return success;
}

}  // namespace mpfem